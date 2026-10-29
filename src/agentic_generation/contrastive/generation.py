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

PROMPT_VERSION = "contrastive-generation-v2"
SYSTEM_PROMPT = """Generate candidate utterance pairs for a speech-to-speech
emotion-adaptation dataset.

The appraisal contrast is supplied by the caller. Use only the supplied
appraisal_axis, appraisal_a, and appraisal_b. Do not choose, rename, reverse,
or invent another contrast.

Core relationship:

Same words + different vocal appraisal -> different appropriate response.

Generate the requested number of distinct candidates. Each candidate must
describe a different event or situation; do not fill the batch with paraphrases
of the same idea.

For every candidate:

1. Write one short, natural spoken English utterance.
2. Make the event clear enough to understand without additional context.
3. Leave the speaker's evaluation of the event unsettled in the text alone.
4. Use exactly one shared utterance for both appraisal conditions.
5. Make both supplied appraisals plausible interpretations of that same event.
6. Do not force an appraisal onto an event when the interpretation would feel
   contrived.
7. Do not include explicit emotion or appraisal statements.
8. Avoid evaluative leakage such as "fortunately", "annoyingly",
   "I'm excited", "I'm worried", "I love that", or "I hate that".
9. Use wording that someone could naturally say in an ordinary conversation.
10. Prefer events for which the broad appraisal contrast could plausibly be
    expressed through vocal delivery.
11. Ensure that the appraisal difference would lead to meaningfully different
    conversational response directions.

Keep the observable event and its outcome identical across both conditions.
Vary the speaker's preferences, goals, or personal stakes to explain the
different appraisals.

Do not create the contrast by inventing different event outcomes, such as
a signed contract in one condition and a rejected contract in the other.

For each condition:

- appraisal must use the supplied appraisal label exactly;
- private_context should briefly explain why the speaker holds that appraisal;
- private_context is acting and generation background, not information
  available to the responding assistant;
- vocal_direction should describe a natural spoken performance that conveys
  the broad appraisal;
- vocal_direction may refer to features such as energy, pace, hesitation,
  emphasis, pitch movement, or vocal tension when useful;
- do not assume that vocal delivery can communicate the exact hidden facts in
  private_context;
- provide at least two distinct response_goals that would be appropriate given
  the utterance and audible appraisal;
- response_goals are alternative response directions, not a sequence of steps;
- response_goals must not assert or depend on private facts unavailable to the
  assistant;
- do not generate exact assistant response sentences.

When writing response_goals, use only the shared utterance and the broad
appraisal that the intended vocal delivery could convey.

Treat private_context as unavailable to the responding assistant.
Do not assert private facts or embed them as assumptions in questions.
When the reason for the appraisal is unknown, invite the speaker to explain it.

Assume a conversational assistant with no physical presence, shared workplace
role, or external-action tools.

Write response_goals as concise descriptions of possible conversational moves.
Do not include quoted response sentences.
Do not assume the assistant can visit, move objects, contact people, or change
schedules. Discussing options or helping the speaker think is allowed.

Before returning a candidate, silently discard it if:

- either appraisal interpretation feels unnatural or contrived;
- the utterance itself reveals which appraisal is intended;
- the two conditions would require different wording;
- the broad contrast could not plausibly be conveyed through spoken delivery;
- both conditions would receive essentially the same response;
- an appropriate response would need to invent or assume hidden facts.

Return only candidates that satisfy these generation constraints.
Do not include quality scores, pass/fail judgments, explanations, or any fields
outside the required structured output.

These are candidate-generation instructions, not proof of quality. Actual
naturalness, tone sufficiency, and response differentiation will be evaluated
separately.
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
