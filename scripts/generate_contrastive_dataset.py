"""Generate resumable contrastive datasets with realized assistant responses."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Annotated

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agentic_generation.config import GenerationSettings
from agentic_generation.contrastive.catalog import CATALOG, CATALOG_VERSION
from agentic_generation.contrastive.generation import (
    PROMPT_VERSION as CANDIDATE_PROMPT_VERSION,
)
from agentic_generation.contrastive.generation import (
    SYSTEM_PROMPT as CANDIDATE_SYSTEM_PROMPT,
)
from agentic_generation.contrastive.generation import (
    ContrastGenerationRequest,
    generate_contrastive_pairs,
)
from agentic_generation.contrastive.models import ResponseGoalCondition, ResponseGoalPair
from agentic_generation.contrastive.response_generation import (
    RESPONSE_GENERATION_PROMPT_VERSION,
    ResponseGenerationRequest,
    ResponseGenerationResult,
    generate_response,
)
from agentic_generation.contrastive.response_generation import (
    SYSTEM_PROMPT as RESPONSE_SYSTEM_PROMPT,
)
from agentic_generation.contrastive.response_goals import (
    RESPONSE_GOALS_PROMPT_VERSION,
    ResponseGoalsRequest,
    generate_response_goals,
)
from agentic_generation.contrastive.response_goals import (
    SYSTEM_PROMPT as RESPONSE_GOALS_SYSTEM_PROMPT,
)
from agentic_generation.contrastive.response_review import (
    RESPONSE_REVIEW_PROMPT_VERSION,
    RESPONSE_REVIEW_SYSTEM_PROMPT,
    build_realized_responses_input,
    determine_response_selection,
    review_realized_responses,
)
from agentic_generation.contrastive.review import (
    CONTRASTIVE_REVIEW_PROMPT_VERSION,
    CONTRASTIVE_REVIEW_SYSTEM_PROMPT,
    RESPONSE_GOALS_REVIEW_PROMPT_VERSION,
    RESPONSE_GOALS_REVIEW_SYSTEM_PROMPT,
    determine_response_goals_decision,
    determine_review_decision,
    review_contrastive_pair,
    review_response_goals,
)

NonEmptyText = Annotated[str, Field(min_length=1)]
DEFAULT_OUTPUT_DIR = Path("results/contrastive")
RESPONSE_SELECTION_POLICY = "reviewer_selected_goal_v1"


class DatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    text: NonEmptyText
    vocal_direction: NonEmptyText


class DatasetCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    private_context: NonEmptyText
    vocal_direction: NonEmptyText
    response_goals: list[NonEmptyText] = Field(min_length=1)
    selected_goal_index: int = Field(ge=0)
    selected_response_goal: NonEmptyText
    response: DatasetResponse

    @model_validator(mode="after")
    def selected_goal_matches_source(self):
        if self.selected_goal_index >= len(self.response_goals):
            raise ValueError("selected_goal_index must reference response_goals")
        if self.selected_response_goal != self.response_goals[self.selected_goal_index]:
            raise ValueError("selected_response_goal must match response_goals")
        return self


class DatasetPair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    pair_id: NonEmptyText
    status: str
    utterance: NonEmptyText
    conditions: dict[NonEmptyText, DatasetCondition]

    @model_validator(mode="after")
    def pair_is_ready_for_audio_review(self):
        if self.status != "ready_for_audio_review":
            raise ValueError("stored pairs must be ready_for_audio_review")
        return self


class AxisDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    status: str
    contrast_id: NonEmptyText
    appraisals: list[NonEmptyText] = Field(min_length=2, max_length=2)
    target_pair_count: int = Field(gt=0)
    completed_pair_count: int = Field(ge=0)
    model: NonEmptyText
    catalog_version: NonEmptyText
    candidate_prompt_version: NonEmptyText
    candidate_prompt_hash: NonEmptyText
    candidate_review_prompt_version: NonEmptyText
    candidate_review_prompt_hash: NonEmptyText
    response_goals_prompt_version: NonEmptyText
    response_goals_prompt_hash: NonEmptyText
    response_goals_review_prompt_version: NonEmptyText
    response_goals_review_prompt_hash: NonEmptyText
    response_prompt_version: NonEmptyText
    response_prompt_hash: NonEmptyText
    response_review_prompt_version: NonEmptyText
    response_review_prompt_hash: NonEmptyText
    response_selection_policy: NonEmptyText
    created_at: NonEmptyText
    updated_at: NonEmptyText
    candidate_calls: int = Field(ge=0)
    candidate_review_calls: int = Field(ge=0)
    response_goal_calls: int = Field(ge=0)
    response_goal_review_calls: int = Field(ge=0)
    response_calls: int = Field(ge=0)
    response_review_calls: int = Field(ge=0)
    pairs: list[DatasetPair]

    @model_validator(mode="after")
    def dataset_is_consistent(self):
        if self.status not in {"in_progress", "completed"}:
            raise ValueError("status must be in_progress or completed")
        if self.completed_pair_count != len(self.pairs):
            raise ValueError("completed_pair_count must match pairs")
        if len({pair.pair_id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("pair_id values must be unique")
        expected = set(self.appraisals)
        if any(set(pair.conditions) != expected for pair in self.pairs):
            raise ValueError("every pair must contain both appraisal conditions")
        if self.response_selection_policy != RESPONSE_SELECTION_POLICY:
            raise ValueError("unsupported response_selection_policy")
        return self


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _dataset_path(output_dir: Path, contrast_id: str) -> Path:
    return output_dir / contrast_id / "dataset.json"


def _new_dataset(
    contrast_id: str,
    target_count: int,
    model: str,
) -> AxisDataset:
    contrast = CATALOG[contrast_id]
    timestamp = _now()
    return AxisDataset(
        status="in_progress",
        contrast_id=contrast_id,
        appraisals=[contrast["appraisal_a"], contrast["appraisal_b"]],
        target_pair_count=target_count,
        completed_pair_count=0,
        model=model,
        catalog_version=CATALOG_VERSION,
        candidate_prompt_version=CANDIDATE_PROMPT_VERSION,
        candidate_prompt_hash=_prompt_hash(CANDIDATE_SYSTEM_PROMPT),
        candidate_review_prompt_version=CONTRASTIVE_REVIEW_PROMPT_VERSION,
        candidate_review_prompt_hash=_prompt_hash(CONTRASTIVE_REVIEW_SYSTEM_PROMPT),
        response_goals_prompt_version=RESPONSE_GOALS_PROMPT_VERSION,
        response_goals_prompt_hash=_prompt_hash(RESPONSE_GOALS_SYSTEM_PROMPT),
        response_goals_review_prompt_version=RESPONSE_GOALS_REVIEW_PROMPT_VERSION,
        response_goals_review_prompt_hash=_prompt_hash(RESPONSE_GOALS_REVIEW_SYSTEM_PROMPT),
        response_prompt_version=RESPONSE_GENERATION_PROMPT_VERSION,
        response_prompt_hash=_prompt_hash(RESPONSE_SYSTEM_PROMPT),
        response_review_prompt_version=RESPONSE_REVIEW_PROMPT_VERSION,
        response_review_prompt_hash=_prompt_hash(RESPONSE_REVIEW_SYSTEM_PROMPT),
        response_selection_policy=RESPONSE_SELECTION_POLICY,
        created_at=timestamp,
        updated_at=timestamp,
        candidate_calls=0,
        candidate_review_calls=0,
        response_goal_calls=0,
        response_goal_review_calls=0,
        response_calls=0,
        response_review_calls=0,
        pairs=[],
    )


def _load_or_create_dataset(
    path: Path,
    contrast_id: str,
    target_count: int,
    model: str,
) -> AxisDataset:
    if not path.exists():
        return _new_dataset(contrast_id, target_count, model)
    stored_data = json.loads(path.read_text(encoding="utf-8"))
    if stored_data.get("response_selection_policy") != RESPONSE_SELECTION_POLICY:
        raise ValueError(
            "existing dataset has no compatible response_selection_policy; "
            "use a new output directory"
        )
    dataset = AxisDataset.model_validate(stored_data)
    expected = _new_dataset(contrast_id, target_count, model)
    immutable_fields = (
        "contrast_id",
        "appraisals",
        "target_pair_count",
        "model",
        "catalog_version",
        "candidate_prompt_version",
        "candidate_prompt_hash",
        "candidate_review_prompt_version",
        "candidate_review_prompt_hash",
        "response_goals_prompt_version",
        "response_goals_prompt_hash",
        "response_goals_review_prompt_version",
        "response_goals_review_prompt_hash",
        "response_prompt_version",
        "response_prompt_hash",
        "response_review_prompt_version",
        "response_review_prompt_hash",
        "response_selection_policy",
    )
    for field_name in immutable_fields:
        if getattr(dataset, field_name) != getattr(expected, field_name):
            raise ValueError(f"existing dataset has incompatible {field_name}")
    return dataset


def _write_dataset(path: Path, dataset: AxisDataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(dataset.model_dump(exclude_none=True), file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _generate_dataset_response(
    client,
    *,
    utterance: str,
    appraisal: str,
    response_goal: str,
    model: str,
    max_output_tokens: int,
) -> ResponseGenerationResult:
    return generate_response(
        client,
        ResponseGenerationRequest(
            utterance=utterance,
            appraisal=appraisal,
            response_goal=response_goal,
        ),
        model,
        max_output_tokens,
    )


def _generate_public_response_goals(
    client,
    *,
    utterance: str,
    appraisal: str,
    model: str,
    max_output_tokens: int,
) -> list[str]:
    result = generate_response_goals(
        client,
        ResponseGoalsRequest(
            utterance=utterance,
            appraisal=appraisal,
        ),
        model,
        max_output_tokens,
    )
    return list(result.response_goals)


def _candidate_passes_review(
    client,
    *,
    pair,
    model: str,
    max_output_tokens: int,
) -> bool:
    review = review_contrastive_pair(
        client,
        pair,
        model,
        max_output_tokens,
    )
    return determine_review_decision(pair, review) == "ready_for_response_goals"


def _realize_pair(
    client,
    pair,
    model: str,
    max_output_tokens: int,
    pair_id: str,
):
    response_goal_calls = 0
    response_goal_review_calls = 0
    response_calls = 0
    response_review_calls = 0
    goal_conditions = []
    for condition in pair.conditions:
        response_goals = _generate_public_response_goals(
            client,
            utterance=pair.utterance,
            appraisal=condition.appraisal,
            model=model,
            max_output_tokens=max_output_tokens,
        )
        response_goal_calls += 1
        goal_conditions.append(
            ResponseGoalCondition(
                appraisal=condition.appraisal,
                private_context=condition.private_context,
                response_goals=response_goals,
            )
        )

    goal_pair = ResponseGoalPair(
        utterance=pair.utterance,
        appraisal_axis=pair.appraisal_axis,
        conditions=goal_conditions,
    )
    goal_review = review_response_goals(client, goal_pair, model, max_output_tokens)
    response_goal_review_calls += 1
    if determine_response_goals_decision(goal_pair, goal_review) != "ready_for_response_generation":
        return (
            None,
            response_goal_calls,
            response_goal_review_calls,
            response_calls,
            response_review_calls,
        )

    response_candidates = []
    for condition in goal_pair.conditions:
        condition_candidates = []
        for response_goal in condition.response_goals:
            response = _generate_dataset_response(
                client,
                utterance=pair.utterance,
                appraisal=condition.appraisal,
                response_goal=response_goal,
                model=model,
                max_output_tokens=max_output_tokens,
            )
            response_calls += 1
            condition_candidates.append(response)
        response_candidates.append(condition_candidates)

    review_input = build_realized_responses_input(goal_pair, response_candidates)
    response_review = review_realized_responses(
        client,
        review_input,
        model,
        max_output_tokens,
    )
    response_review_calls += 1
    response_decision, selected_indexes = determine_response_selection(
        review_input,
        response_review,
    )
    if response_decision != "ready_for_audio_review" or selected_indexes is None:
        return (
            None,
            response_goal_calls,
            response_goal_review_calls,
            response_calls,
            response_review_calls,
        )

    original_conditions = {condition.appraisal: condition for condition in pair.conditions}
    conditions = {}
    for condition_index, condition in enumerate(goal_pair.conditions):
        selected_goal_index = selected_indexes[condition_index]
        selected_response = response_candidates[condition_index][selected_goal_index]
        original_condition = original_conditions[condition.appraisal]
        conditions[condition.appraisal] = DatasetCondition(
            private_context=original_condition.private_context,
            vocal_direction=original_condition.vocal_direction,
            response_goals=condition.response_goals,
            selected_goal_index=selected_goal_index,
            selected_response_goal=condition.response_goals[selected_goal_index],
            response=DatasetResponse(
                text=selected_response.response_text,
                vocal_direction=selected_response.response_vocal_direction,
            ),
        )
    return (
        DatasetPair(
            pair_id=pair_id,
            status="ready_for_audio_review",
            utterance=pair.utterance,
            conditions=conditions,
        ),
        response_goal_calls,
        response_goal_review_calls,
        response_calls,
        response_review_calls,
    )


def generate_axis(
    client,
    *,
    contrast_id: str,
    target_count: int,
    batch_size: int,
    max_batches: int,
    output_dir: Path,
    model: str,
    max_output_tokens: int,
) -> AxisDataset:
    """Generate one axis, checkpointing each fully realized pair."""
    path = _dataset_path(output_dir, contrast_id)
    dataset = _load_or_create_dataset(path, contrast_id, target_count, model)
    if dataset.completed_pair_count >= target_count:
        return dataset

    seen_utterances = {_normalize(pair.utterance) for pair in dataset.pairs}
    batches_this_run = 0
    while dataset.completed_pair_count < target_count:
        if batches_this_run >= max_batches:
            raise RuntimeError(
                f"{contrast_id} did not reach {target_count} unique pairs "
                f"within {max_batches} batches"
            )
        requested_count = min(batch_size, target_count - dataset.completed_pair_count)
        batch = generate_contrastive_pairs(
            client,
            ContrastGenerationRequest(
                contrast_id=contrast_id,
                candidate_count=requested_count,
            ),
            model,
            max_output_tokens,
        )
        batches_this_run += 1
        dataset.candidate_calls += 1

        for candidate in batch.pairs:
            normalized = _normalize(candidate.utterance)
            if normalized in seen_utterances:
                continue
            candidate_passed = _candidate_passes_review(
                client,
                pair=candidate,
                model=model,
                max_output_tokens=max_output_tokens,
            )
            dataset.candidate_review_calls += 1
            if not candidate_passed:
                continue
            pair_id = f"{contrast_id}__{dataset.completed_pair_count + 1:03d}"
            (
                realized,
                response_goal_calls,
                response_goal_review_calls,
                response_calls,
                response_review_calls,
            ) = _realize_pair(
                client,
                candidate,
                model,
                max_output_tokens,
                pair_id,
            )
            dataset.response_goal_calls += response_goal_calls
            dataset.response_goal_review_calls += response_goal_review_calls
            dataset.response_calls += response_calls
            dataset.response_review_calls += response_review_calls
            if realized is None:
                continue
            dataset.pairs.append(realized)
            seen_utterances.add(normalized)
            dataset.completed_pair_count = len(dataset.pairs)
            dataset.updated_at = _now()
            if dataset.completed_pair_count == target_count:
                dataset.status = "completed"
            _write_dataset(path, dataset)

    return dataset


def build_dry_run(
    contrast_ids: list[str],
    target_count: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "status": "dry_run",
        "response_selection_policy": RESPONSE_SELECTION_POLICY,
        "contrast_ids": contrast_ids,
        "target_pairs_per_axis": target_count,
        "total_target_pairs": target_count * len(contrast_ids),
        "minimum_candidate_calls": ceil(target_count / batch_size) * len(contrast_ids),
        "minimum_candidate_review_calls": target_count * len(contrast_ids),
        "response_goal_calls": target_count * 2 * len(contrast_ids),
        "response_goal_review_calls": target_count * len(contrast_ids),
        "response_calls": target_count * 4 * len(contrast_ids),
        "response_review_calls": target_count * len(contrast_ids),
        "output_files": [
            str(_dataset_path(output_dir, contrast_id)) for contrast_id in contrast_ids
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one resumable JSON dataset per appraisal axis."
    )
    parser.add_argument(
        "--contrast-id",
        action="append",
        choices=tuple(CATALOG),
        help="Generate only this axis; repeat for multiple axes. Defaults to all axes.",
    )
    parser.add_argument("--count-per-axis", type=int, default=50)
    parser.add_argument("--batch-size", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--max-batches-per-axis", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count_per_axis < 1 or args.max_batches_per_axis < 1:
        print("Counts must be positive integers.", file=sys.stderr)
        return 2
    contrast_ids = args.contrast_id or list(CATALOG)
    if args.dry_run:
        print(
            json.dumps(
                build_dry_run(
                    contrast_ids,
                    args.count_per_axis,
                    args.batch_size,
                    args.output_dir,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    try:
        settings = GenerationSettings()
    except ValidationError:
        print("Invalid OpenAI settings. Check OPENAI_* values in .env.", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=settings.api_key.get_secret_value(),
        timeout=60.0,
        max_retries=0,
    )
    try:
        for contrast_id in contrast_ids:
            dataset = generate_axis(
                client,
                contrast_id=contrast_id,
                target_count=args.count_per_axis,
                batch_size=args.batch_size,
                max_batches=args.max_batches_per_axis,
                output_dir=args.output_dir,
                model=settings.user_model,
                max_output_tokens=settings.max_output_tokens,
            )
            print(
                f"{contrast_id}: {dataset.completed_pair_count}/{dataset.target_pair_count} pairs"
            )
    except (KeyboardInterrupt, Exception) as error:
        if isinstance(error, KeyboardInterrupt):
            print("Generation interrupted; completed pairs were checkpointed.", file=sys.stderr)
        else:
            print(f"Generation stopped: {error}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
