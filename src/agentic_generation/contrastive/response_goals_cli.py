"""Regenerate and compare response goals from a stored contrastive artifact."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentic_generation.config import GenerationSettings
from agentic_generation.contrastive.models import ContrastivePair
from agentic_generation.contrastive.response_goals import (
    RESPONSE_GOALS_PROMPT_VERSION,
    SYSTEM_PROMPT,
    ResponseGoalsRequest,
    generate_response_goals,
)
from agentic_generation.openai_adapter import GenerationError

NonEmptyText = Annotated[str, Field(min_length=1)]


class _SourceRun(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, str_strip_whitespace=True)

    id: NonEmptyText


class _SourceArtifact(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, str_strip_whitespace=True)

    status: Literal["completed"]
    run: _SourceRun
    pairs: list[ContrastivePair] = Field(min_length=1)


def _load_source_artifact(path: Path) -> _SourceArtifact | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _SourceArtifact.model_validate(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        print(
            "Invalid contrastive artifact. Check the file and its completed pairs.",
            file=sys.stderr,
        )
        return None


def _print_output(
    base_output: dict[str, object],
    status: Literal["completed", "failed"],
    comparisons: list[dict[str, object]],
    failure: dict[str, object] | None,
) -> None:
    print(
        json.dumps(
            {
                **base_output,
                "status": status,
                "completed_conditions": len(comparisons),
                "comparisons": comparisons,
                "failure": failure,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def regenerate_response_goals(path: Path) -> int:
    """Regenerate goals for every stored condition and print a comparison artifact."""
    artifact = _load_source_artifact(path)
    if artifact is None:
        return 2

    try:
        settings = GenerationSettings()
    except ValidationError:
        print(
            "Invalid OpenAI settings. Check OPENAI_* values in .env.",
            file=sys.stderr,
        )
        return 2

    requested_conditions = sum(len(pair.conditions) for pair in artifact.pairs)
    base_output: dict[str, object] = {
        "review_status": "unreviewed",
        "run": {
            "id": str(uuid4()),
            "created_at": datetime.now(UTC).isoformat(),
        },
        "source_run_id": artifact.run.id,
        "model": settings.user_model,
        "generation_settings": {
            "max_output_tokens": settings.max_output_tokens,
            "timeout_seconds": 60.0,
            "max_retries": 0,
        },
        "response_goals_prompt_version": RESPONSE_GOALS_PROMPT_VERSION,
        "response_goals_prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "requested_conditions": requested_conditions,
    }
    comparisons: list[dict[str, object]] = []
    client = None
    pair_index: int | None = None
    condition_index: int | None = None
    failure_reason: str | None = None
    failure_message: str | None = None

    try:
        client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=60.0,
            max_retries=0,
        )
        for pair_index, pair in enumerate(artifact.pairs):
            for condition_index, condition in enumerate(pair.conditions):
                request = ResponseGoalsRequest(
                    utterance=pair.utterance,
                    appraisal=condition.appraisal,
                )
                result = generate_response_goals(
                    client,
                    request,
                    settings.user_model,
                    settings.max_output_tokens,
                )
                comparisons.append(
                    {
                        "pair_index": pair_index,
                        "condition_index": condition_index,
                        "appraisal_axis": pair.appraisal_axis,
                        "utterance": pair.utterance,
                        "appraisal": condition.appraisal,
                        "original_response_goals": condition.response_goals,
                        "new_response_goals": result.response_goals,
                    }
                )
    except GenerationError:
        failure_reason = "invalid_generation"
        failure_message = "Response-goal generation returned no valid result."
    except APITimeoutError:
        failure_reason = "timeout"
        failure_message = "Response-goal generation timed out."
    except APIConnectionError:
        failure_reason = "connection_error"
        failure_message = "Could not connect to the response-goal provider."
    except APIStatusError as error:
        failure_reason = f"http_{error.status_code}"
        failure_message = f"Response-goal provider returned HTTP {error.status_code}."
    except Exception:
        failure_reason = "unexpected_error"
        failure_message = "Response-goal generation failed unexpectedly."
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    if failure_reason is not None:
        _print_output(
            base_output,
            "failed",
            comparisons,
            {
                "pair_index": pair_index,
                "condition_index": condition_index,
                "reason": failure_reason,
            },
        )
        print(failure_message, file=sys.stderr)
        return 1

    _print_output(base_output, "completed", comparisons, None)
    return 0
