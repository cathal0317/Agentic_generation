"""Generate candidate pairs for one catalog-selected appraisal contrast."""

import json
from typing import Annotated

from openai import OpenAI
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from agentic_generation.contrastive.catalog import CATALOG
from agentic_generation.contrastive.models import ContrastivePair
from agentic_generation.openai_adapter import GenerationError

PROMPT_VERSION = "contrastive-generation-v1"
SYSTEM_PROMPT = """Generate contrastive utterance-pair candidates using only the supplied axis.
Use the supplied appraisal axis and both appraisal labels exactly as provided.
Generate the requested number of distinct events and situations; do not fill the batch with
simple paraphrases.
Write short, natural English utterances whose events are clear but whose appraisal is not settled.
Do not state the speaker's emotion or appraisal directly in the utterance.
The same shared utterance must be natural under both appraisals.
Keep the event itself unchanged between conditions; change the speaker's appraisal of it.
Treat private_context as private generation background, not information available to an assistant.
Use vocal_direction to guide a natural spoken performance.
Do not assume that a voice can communicate specific hidden facts.
For each condition, propose at least two distinct possible response_goals.
Treat response_goals as alternative response directions, not steps that must all be performed.
Every response direction must make sense without already knowing the private background.
The appropriate response directions must differ meaningfully between the two appraisals.
Do not generate exact assistant responses or self-evaluations.

Core relationship:
Same words + different vocal appraisal -> different appropriate response.
"""
NonEmptyText = Annotated[str, Field(min_length=1)]


class ContrastGenerationRequest(BaseModel):
    """A request for candidates from one fixed catalog contrast."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    contrast_id: NonEmptyText
    candidate_count: int = Field(ge=1, le=6)

    @field_validator("contrast_id")
    @classmethod
    def contrast_exists_in_catalog(cls, value: str) -> str:
        if value not in CATALOG:
            raise ValueError("unknown contrast_id")
        return value


class ContrastiveBatch(BaseModel):
    """Structurally valid contrastive candidates returned by the provider."""

    model_config = ConfigDict(extra="forbid", strict=True)

    pairs: list[ContrastivePair] = Field(min_length=1, max_length=6)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def generate_contrastive_pairs(
    client: OpenAI,
    request: ContrastGenerationRequest,
    model: str,
    max_output_tokens: int,
) -> ContrastiveBatch:
    """Generate and deterministically validate one requested candidate batch."""
    contrast = CATALOG[request.contrast_id]
    provider_input = {
        "appraisal_axis": request.contrast_id,
        "appraisal_a": contrast["appraisal_a"],
        "appraisal_b": contrast["appraisal_b"],
        "description": contrast["description"],
        "candidate_count": request.candidate_count,
    }

    try:
        response = client.responses.parse(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": json.dumps(provider_input, ensure_ascii=False),
                }
            ],
            text_format=ContrastiveBatch,
            store=False,
            max_output_tokens=max_output_tokens,
        )
    except ValidationError as error:
        raise GenerationError(
            "Contrastive generation returned invalid structured output"
        ) from error

    if response.status != "completed" or not isinstance(response.output_parsed, ContrastiveBatch):
        raise GenerationError("No complete, valid contrastive batch was returned")

    batch = response.output_parsed
    if len(batch.pairs) != request.candidate_count:
        raise GenerationError("Contrastive batch candidate count did not match the request")

    expected_appraisals = {
        _normalize(contrast["appraisal_a"]),
        _normalize(contrast["appraisal_b"]),
    }
    for pair in batch.pairs:
        if pair.appraisal_axis != request.contrast_id:
            raise GenerationError("Contrastive pair appraisal axis did not match the request")
        actual_appraisals = {_normalize(condition.appraisal) for condition in pair.conditions}
        if actual_appraisals != expected_appraisals:
            raise GenerationError("Contrastive pair appraisals did not match the catalog")

    normalized_utterances = [_normalize(pair.utterance) for pair in batch.pairs]
    if len(set(normalized_utterances)) != len(normalized_utterances):
        raise GenerationError("Contrastive batch contains duplicate utterances")

    return batch
