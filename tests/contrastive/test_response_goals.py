import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_generation.contrastive.response_goals import (
    ResponseGoalsRequest,
    ResponseGoalsResult,
    generate_response_goals,
)
from agentic_generation.openai_adapter import GenerationError

PRIVATE_MARKER = "PRIVATE_CONFLICT_K9Q7"


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


@pytest.fixture
def goals_request():
    return ResponseGoalsRequest(
        utterance="They moved the meeting to Friday.",
        appraisal="unwanted",
    )


@pytest.fixture
def result():
    return ResponseGoalsResult(
        response_goals=[
            "Acknowledge that the change seems unwelcome.",
            "Invite the speaker to explain what concerns them about it.",
        ]
    )


def test_returns_valid_result(goals_request, result):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=result))

    generated = generate_response_goals(FakeClient(responses), goals_request, "model", 256)

    assert generated == result
    assert ResponseGoalsResult.model_validate_json(generated.model_dump_json()) == result


def test_sends_only_utterance_and_appraisal_in_one_structured_request(goals_request, result):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=result))

    generate_response_goals(FakeClient(responses), goals_request, "response-model", 384)

    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["model"] == "response-model"
    assert call["max_output_tokens"] == 384
    assert call["text_format"] is ResponseGoalsResult
    assert call["store"] is False
    assert json.loads(call["input"][0]["content"]) == {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
    }
    assert PRIVATE_MARKER not in repr(call["input"])
    assert responses.create_calls == []


@pytest.mark.parametrize(
    "extra_field",
    ["private_context", "vocal_direction", "response_goals"],
)
def test_request_rejects_condition_details(extra_field):
    data = {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
        extra_field: PRIVATE_MARKER,
    }

    with pytest.raises(ValidationError):
        ResponseGoalsRequest.model_validate(data)


@pytest.mark.parametrize("field", ["utterance", "appraisal"])
def test_request_rejects_blank_text(field):
    data = {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
    }
    data[field] = " \n\t "

    with pytest.raises(ValidationError):
        ResponseGoalsRequest.model_validate(data)


@pytest.mark.parametrize(
    "goals",
    [
        [],
        ["One goal."],
        ["One goal.", "Two goals.", "Three goals."],
        ["One goal.", "   "],
    ],
)
def test_result_requires_exactly_two_non_empty_goals(goals):
    with pytest.raises(ValidationError):
        ResponseGoalsResult.model_validate({"response_goals": goals})


def test_result_rejects_normalized_duplicate_goals():
    with pytest.raises(ValidationError):
        ResponseGoalsResult.model_validate(
            {
                "response_goals": [
                    " Invite the speaker to explain. ",
                    "INVITE   THE SPEAKER TO EXPLAIN.",
                ]
            }
        )


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_rejects_non_completed_response(goals_request, result, status):
    responses = FakeResponses(SimpleNamespace(status=status, output_parsed=result))

    with pytest.raises(GenerationError, match="No complete"):
        generate_response_goals(FakeClient(responses), goals_request, "model", 256)


def test_rejects_missing_parsed_output(goals_request):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=None))

    with pytest.raises(GenerationError, match="No complete"):
        generate_response_goals(FakeClient(responses), goals_request, "model", 256)


def test_converts_parse_validation_error_safely(goals_request):
    try:
        ResponseGoalsResult.model_validate({"response_goals": ["only one"]})
    except ValidationError as error:
        parse_error = error
    responses = FakeResponses(error=parse_error)

    with pytest.raises(GenerationError, match="invalid structured output") as caught_error:
        generate_response_goals(FakeClient(responses), goals_request, "model", 256)

    assert "only one" not in str(caught_error.value)
    assert len(responses.parse_calls) == 1
    assert responses.create_calls == []


def test_provider_error_propagates_without_retry_or_fallback(goals_request):
    provider_error = RuntimeError("provider failed")
    responses = FakeResponses(error=provider_error)

    with pytest.raises(RuntimeError, match="provider failed") as caught_error:
        generate_response_goals(FakeClient(responses), goals_request, "model", 256)

    assert caught_error.value is provider_error
    assert len(responses.parse_calls) == 1
    assert responses.create_calls == []
