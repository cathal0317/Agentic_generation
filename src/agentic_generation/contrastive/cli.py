"""CLI handler for contrastive appraisal candidate generation."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from uuid import uuid4

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError

from agentic_generation.config import GenerationSettings
from agentic_generation.contrastive.catalog import CATALOG_VERSION
from agentic_generation.contrastive.generation import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ContrastGenerationRequest,
    generate_contrastive_pairs,
)
from agentic_generation.openai_adapter import GenerationError


def generate_contrastive(contrast_id: str, count: int) -> int:
    """Validate, generate, and print one contrastive candidate batch."""
    try:
        request = ContrastGenerationRequest(
            contrast_id=contrast_id,
            candidate_count=count,
        )
    except ValidationError:
        print(
            "Invalid contrastive request. Check --contrast-id and --count.",
            file=sys.stderr,
        )
        return 2

    try:
        settings = GenerationSettings()
    except ValidationError:
        print(
            "Invalid OpenAI settings. Check OPENAI_* values in .env.",
            file=sys.stderr,
        )
        return 2

    client = None
    try:
        client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=60.0,
            max_retries=0,
        )
        batch = generate_contrastive_pairs(
            client,
            request,
            settings.user_model,
            settings.max_output_tokens,
        )
    except GenerationError as error:
        print(f"Contrastive generation failed: {error}", file=sys.stderr)
        return 1
    except APITimeoutError:
        print("Contrastive generation failed: OpenAI request timed out.", file=sys.stderr)
        return 1
    except APIConnectionError:
        print(
            "Contrastive generation failed: could not connect to OpenAI.",
            file=sys.stderr,
        )
        return 1
    except APIStatusError as error:
        print(
            f"Contrastive generation failed: OpenAI returned HTTP {error.status_code}.",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            "Contrastive generation failed. Check OpenAI configuration and availability.",
            file=sys.stderr,
        )
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    print(
        json.dumps(
            {
                "status": "completed",
                "review_status": "unreviewed",
                "run": {
                    "id": str(uuid4()),
                    "created_at": datetime.now(UTC).isoformat(),
                },
                "request": request.model_dump(),
                "model": settings.user_model,
                "generation_settings": {
                    "max_output_tokens": settings.max_output_tokens,
                    "timeout_seconds": 60.0,
                    "max_retries": 0,
                },
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                "catalog_version": CATALOG_VERSION,
                "pairs": [pair.model_dump() for pair in batch.pairs],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0
