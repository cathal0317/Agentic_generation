import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentic_generation.contrastive.response_generation import (
    RESPONSE_GENERATION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    ResponseGenerationRequest,
    ResponseGenerationResult,
    generate_response,
)
from agentic_generation.openai_adapter import GenerationError


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
def generation_request():
    return ResponseGenerationRequest(
        utterance="They moved the meeting to Friday.",
        appraisal="unwanted",
        response_goal="Briefly acknowledge that the change seems unwelcome.",
    )


@pytest.fixture
def generation_result():
    return ResponseGenerationResult(
        response_text="That change doesn't sound especially welcome.",
        response_vocal_direction="Calm and attentive, with gentle emphasis on 'change'.",
    )


def test_request_and_result_strip_surrounding_whitespace():
    request = ResponseGenerationRequest(
        utterance="  They moved the meeting to Friday. \n",
        appraisal="\t unwanted ",
        response_goal=" Briefly acknowledge the reaction. ",
    )
    result = ResponseGenerationResult(
        response_text="  That sounds unwelcome. ",
        response_vocal_direction="\n Calm and attentive. ",
    )

    assert request.model_dump() == {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
        "response_goal": "Briefly acknowledge the reaction.",
    }
    assert result.model_dump() == {
        "response_text": "That sounds unwelcome.",
        "response_vocal_direction": "Calm and attentive.",
    }


def test_request_allows_omitted_or_none_response_goal():
    omitted = ResponseGenerationRequest(
        utterance="They moved the meeting to Friday.",
        appraisal="wanted",
    )
    explicit_none = ResponseGenerationRequest(
        utterance="They moved the meeting to Friday.",
        appraisal="wanted",
        response_goal=None,
    )

    assert omitted.response_goal is None
    assert explicit_none.response_goal is None


@pytest.mark.parametrize(
    "field",
    ["utterance", "appraisal", "response_goal"],
)
@pytest.mark.parametrize("invalid_value", ["", " \n\t"])
def test_request_rejects_empty_or_blank_text(field, invalid_value):
    data = {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
        "response_goal": "Acknowledge the reaction.",
    }
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        ResponseGenerationRequest.model_validate(data)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("utterance", 123),
        ("appraisal", ["unwanted"]),
        ("response_goal", {"move": "acknowledge"}),
    ],
)
def test_request_rejects_wrong_types(field, invalid_value):
    data = {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
        "response_goal": "Acknowledge the reaction.",
    }
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        ResponseGenerationRequest.model_validate(data)


@pytest.mark.parametrize(
    "extra_field",
    [
        "private_context",
        "vocal_direction",
        "response_goals",
        "opposite_condition",
        "review",
    ],
)
def test_request_rejects_extra_condition_and_review_fields(extra_field):
    data = {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
        "response_goal": "Acknowledge the reaction.",
        extra_field: "PRIVATE_MARKER_X7",
    }

    with pytest.raises(ValidationError):
        ResponseGenerationRequest.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {
            "response_text": " ",
            "response_vocal_direction": "Calm.",
        },
        {
            "response_text": "That sounds unwelcome.",
            "response_vocal_direction": "",
        },
        {
            "response_text": 123,
            "response_vocal_direction": "Calm.",
        },
        {
            "response_text": "That sounds unwelcome.",
            "response_vocal_direction": ["Calm."],
        },
        {
            "response_text": "That sounds unwelcome.",
            "response_vocal_direction": "Calm.",
            "score": 1,
        },
    ],
)
def test_result_rejects_blank_wrong_type_and_extra_fields(data):
    with pytest.raises(ValidationError):
        ResponseGenerationResult.model_validate(data)


def test_sends_only_allowed_payload_in_one_structured_request(
    generation_request,
    generation_result,
):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=generation_result))
    client = FakeClient(responses)
    original_request = generation_request.model_dump()

    result = generate_response(
        client,
        generation_request,
        "response-model",
        384,
    )

    assert result == generation_result
    assert generation_request.model_dump() == original_request
    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["model"] == "response-model"
    assert call["max_output_tokens"] == 384
    assert call["text_format"] is ResponseGenerationResult
    assert call["store"] is False
    assert json.loads(call["input"][0]["content"]) == {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "unwanted",
        "response_goal": "Briefly acknowledge that the change seems unwelcome.",
    }
    assert responses.create_calls == []


@pytest.mark.parametrize("goal_argument", [{}, {"response_goal": None}])
def test_omits_absent_goal_from_provider_payload(generation_result, goal_argument):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=generation_result))
    request = ResponseGenerationRequest(
        utterance="They moved the meeting to Friday.",
        appraisal="wanted",
        **goal_argument,
    )

    generate_response(FakeClient(responses), request, "response-model", 384)

    assert len(responses.parse_calls) == 1
    assert json.loads(responses.parse_calls[0]["input"][0]["content"]) == {
        "utterance": "They moved the meeting to Friday.",
        "appraisal": "wanted",
    }
    assert responses.create_calls == []


def test_v2_prompt_contains_shared_goal_and_no_goal_rules():
    assert RESPONSE_GENERATION_PROMPT_VERSION == "response-generation-v2"
    assert "If a response_goal is provided" in SYSTEM_PROMPT
    assert "If no response_goal is provided" in SYSTEM_PROMPT
    assert "Do not invent the cause of the appraisal" in SYSTEM_PROMPT
    assert "Acknowledgment does not require reassurance, advice, or a positive reframing." in (
        SYSTEM_PROMPT
    )
    assert "prior assistant actions" in SYSTEM_PROMPT


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_rejects_non_completed_response(
    generation_request,
    generation_result,
    status,
):
    responses = FakeResponses(SimpleNamespace(status=status, output_parsed=generation_result))

    with pytest.raises(GenerationError, match="No complete"):
        generate_response(FakeClient(responses), generation_request, "model", 256)


@pytest.mark.parametrize(
    "output_parsed",
    [
        None,
        ResponseGenerationRequest(
            utterance="They moved the meeting to Friday.",
            appraisal="unwanted",
            response_goal="Acknowledge the reaction.",
        ),
    ],
)
def test_rejects_missing_or_wrong_parsed_output(generation_request, output_parsed):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=output_parsed))

    with pytest.raises(GenerationError, match="No complete"):
        generate_response(FakeClient(responses), generation_request, "model", 256)


def test_converts_parse_validation_error_safely(generation_request):
    try:
        ResponseGenerationResult.model_validate(
            {
                "response_text": "PRIVATE_INPUT_M4",
                "response_vocal_direction": " ",
            }
        )
    except ValidationError as error:
        parse_error = error
    responses = FakeResponses(error=parse_error)

    with pytest.raises(GenerationError, match="invalid structured output") as caught_error:
        generate_response(FakeClient(responses), generation_request, "model", 256)

    assert "PRIVATE_INPUT_M4" not in str(caught_error.value)
    assert len(responses.parse_calls) == 1
    assert responses.create_calls == []


def test_transport_error_propagates_without_retry_or_fallback(generation_request):
    provider_error = RuntimeError("transport failed")
    responses = FakeResponses(error=provider_error)

    with pytest.raises(RuntimeError, match="transport failed") as caught_error:
        generate_response(FakeClient(responses), generation_request, "model", 256)

    assert caught_error.value is provider_error
    assert len(responses.parse_calls) == 1
    assert responses.create_calls == []
