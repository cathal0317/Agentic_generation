import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_generation.contrastive.catalog import CATALOG, CATALOG_VERSION
from agentic_generation.contrastive.generation import (
    ContrastGenerationRequest,
    ContrastiveBatch,
    generate_contrastive_pairs,
)
from agentic_generation.contrastive.models import ContrastCondition, ContrastivePair
from agentic_generation.openai_adapter import GenerationError

EXPECTED_LABELS = {
    "wanted_vs_unwanted": ("wanted", "unwanted"),
    "opportunity_vs_burden": ("opportunity", "burden"),
    "relief_vs_disruption": ("relief", "disruption"),
    "anticipation_vs_anxiety": ("anticipation", "anxiety"),
    "pride_vs_embarrassment": ("pride", "embarrassment"),
    "welcome_vs_intrusion": ("welcome", "intrusion"),
    "success_vs_disappointment": ("success", "disappointment"),
    "amusement_vs_offence": ("amusement", "offence"),
}


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.parse_calls = []
        self.create_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        raise AssertionError("unstructured fallback must not run")


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


def make_pair(
    utterance="They put me in charge.",
    axis="opportunity_vs_burden",
    appraisals=("opportunity", "burden"),
):
    return ContrastivePair(
        utterance=utterance,
        appraisal_axis=axis,
        conditions=[
            ContrastCondition(
                appraisal=appraisals[0],
                private_context="The speaker wanted the responsibility.",
                vocal_direction="Quietly pleased.",
                response_goals=["Acknowledge the opportunity.", "Ask what appeals to them."],
            ),
            ContrastCondition(
                appraisal=appraisals[1],
                private_context="The speaker did not want more responsibility.",
                vocal_direction="Tired and weighed down.",
                response_goals=["Acknowledge the pressure.", "Ask what support is available."],
            ),
        ],
    )


def completed_response(batch):
    return SimpleNamespace(status="completed", output_parsed=batch)


def test_catalog_contains_only_the_eight_specified_label_pairs():
    assert CATALOG_VERSION == "appraisal-catalog-v1"
    assert set(CATALOG) == set(EXPECTED_LABELS)
    assert {
        contrast_id: (entry["appraisal_a"], entry["appraisal_b"])
        for contrast_id, entry in CATALOG.items()
    } == EXPECTED_LABELS
    assert all(
        set(entry) == {"appraisal_a", "appraisal_b", "description"} for entry in CATALOG.values()
    )


@pytest.mark.parametrize("contrast_id", EXPECTED_LABELS)
def test_every_catalog_id_is_a_valid_request(contrast_id):
    request = ContrastGenerationRequest(contrast_id=contrast_id, candidate_count=1)

    assert request.contrast_id == contrast_id


@pytest.mark.parametrize(
    "data",
    [
        {"contrast_id": "not_in_catalog", "candidate_count": 1},
        {"contrast_id": "   ", "candidate_count": 1},
        {"contrast_id": "opportunity_vs_burden", "candidate_count": 0},
        {"contrast_id": "opportunity_vs_burden", "candidate_count": 7},
        {
            "contrast_id": "opportunity_vs_burden",
            "candidate_count": 1,
            "appraisal_a": "opportunity",
        },
    ],
)
def test_request_rejects_invalid_values_and_extra_fields(data):
    with pytest.raises(ValidationError):
        ContrastGenerationRequest.model_validate(data)


def test_returns_valid_batch_with_json_round_trip():
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)
    batch = ContrastiveBatch(pairs=[make_pair()])
    responses = FakeResponses(completed_response(batch))

    result = generate_contrastive_pairs(FakeClient(responses), request, "model", 512)

    assert result == batch
    assert ContrastiveBatch.model_validate_json(result.model_dump_json()) == result


def test_sends_only_selected_contrast_and_count_in_one_structured_request():
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)
    batch = ContrastiveBatch(pairs=[make_pair()])
    responses = FakeResponses(completed_response(batch))

    generate_contrastive_pairs(FakeClient(responses), request, "user-model", 768)

    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["model"] == "user-model"
    assert call["max_output_tokens"] == 768
    assert call["text_format"] is ContrastiveBatch
    assert call["store"] is False
    assert json.loads(call["input"][0]["content"]) == {
        "appraisal_axis": "opportunity_vs_burden",
        **CATALOG["opportunity_vs_burden"],
        "candidate_count": 1,
    }
    assert responses.create_calls == []


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_rejects_non_completed_responses(status):
    batch = ContrastiveBatch(pairs=[make_pair()])
    responses = FakeResponses(SimpleNamespace(status=status, output_parsed=batch))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    with pytest.raises(GenerationError, match="No complete"):
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)


def test_rejects_missing_parsed_output():
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=None))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    with pytest.raises(GenerationError, match="No complete"):
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)


def test_rejects_wrong_candidate_count():
    batch = ContrastiveBatch(pairs=[make_pair()])
    responses = FakeResponses(completed_response(batch))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=2)

    with pytest.raises(GenerationError, match="candidate count"):
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)


def test_rejects_wrong_appraisal_axis():
    batch = ContrastiveBatch(pairs=[make_pair(axis="wanted_vs_unwanted")])
    responses = FakeResponses(completed_response(batch))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    with pytest.raises(GenerationError, match="appraisal axis"):
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)


def test_rejects_wrong_appraisals():
    batch = ContrastiveBatch(pairs=[make_pair(appraisals=("opportunity", "disappointment"))])
    responses = FakeResponses(completed_response(batch))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    with pytest.raises(GenerationError, match="appraisals"):
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)


def test_accepts_reversed_condition_order_and_normalized_appraisals():
    batch = ContrastiveBatch(pairs=[make_pair(appraisals=("  BURDEN  ", " opportunity "))])
    responses = FakeResponses(completed_response(batch))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    result = generate_contrastive_pairs(FakeClient(responses), request, "model", 512)

    assert result == batch


def test_rejects_normalized_duplicate_utterances():
    batch = ContrastiveBatch(
        pairs=[
            make_pair(utterance="They put me in charge."),
            make_pair(utterance="  THEY put me   in charge.  "),
        ]
    )
    responses = FakeResponses(completed_response(batch))
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=2)

    with pytest.raises(GenerationError, match="duplicate utterances"):
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)


def test_converts_parse_validation_error_without_exposing_details():
    try:
        ContrastiveBatch.model_validate({"pairs": []})
    except ValidationError as error:
        parse_error = error
    responses = FakeResponses(error=parse_error)
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    with pytest.raises(GenerationError, match="invalid structured output") as caught_error:
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)

    assert "pairs" not in str(caught_error.value)
    assert len(responses.parse_calls) == 1
    assert responses.create_calls == []


def test_propagates_provider_errors_without_retry_or_fallback():
    provider_error = RuntimeError("provider failed")
    responses = FakeResponses(error=provider_error)
    request = ContrastGenerationRequest(contrast_id="opportunity_vs_burden", candidate_count=1)

    with pytest.raises(RuntimeError, match="provider failed") as caught_error:
        generate_contrastive_pairs(FakeClient(responses), request, "model", 512)

    assert caught_error.value is provider_error
    assert len(responses.parse_calls) == 1
    assert responses.create_calls == []
