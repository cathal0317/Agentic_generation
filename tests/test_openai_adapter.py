import json
from types import SimpleNamespace

import pytest

from agentic_generation.controller import REVIEWER_SYSTEM_PROMPT, DialogueAssessment
from agentic_generation.openai_adapter import (
    GenerationError,
    create_dialogue_reviewer,
    create_text_generator,
)


class FakeResponses:
    def __init__(self, response=None, error=None, parsed_response=None, parse_error=None):
        self.response = response
        self.error = error
        self.parsed_response = parsed_response
        self.parse_error = parse_error
        self.calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if self.parse_error is not None:
            raise self.parse_error
        return self.parsed_response


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


def test_generator_sends_complete_request_with_preserved_messages():
    responses = FakeResponses(SimpleNamespace(status="completed", output_text="Generated text"))
    generator = create_text_generator(FakeClient(responses), "requested-model", 321)
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
    ]

    assert generator("System instructions", messages) == "Generated text"
    assert responses.calls == [
        {
            "model": "requested-model",
            "instructions": "System instructions",
            "input": messages,
            "store": False,
            "max_output_tokens": 321,
        }
    ]


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status="incomplete", output_text="Partial text"),
        SimpleNamespace(status="completed", output_text=""),
        SimpleNamespace(status="completed", output_text=" \n "),
    ],
)
def test_generator_rejects_incomplete_or_empty_responses(response):
    generator = create_text_generator(FakeClient(FakeResponses(response)), "model", 2048)

    with pytest.raises(GenerationError):
        generator("Instructions", [{"role": "user", "content": "Hello"}])


def test_generator_propagates_provider_errors():
    provider_error = RuntimeError("provider failed")
    generator = create_text_generator(
        FakeClient(FakeResponses(error=provider_error)),
        "model",
        2048,
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        generator("Instructions", [{"role": "user", "content": "Hello"}])


def test_reviewer_sends_only_the_public_transcript_for_structured_parsing():
    assessment = DialogueAssessment(
        decision="continue",
        reason="The assistant answered a new question.",
    )
    responses = FakeResponses(
        parsed_response=SimpleNamespace(
            status="completed",
            output_parsed=assessment,
        )
    )
    reviewer = create_dialogue_reviewer(FakeClient(responses), "assistant-model", 456)
    transcript = [
        {"turn_index": 0, "role": "user", "text": "Can we discuss this?"},
        {"turn_index": 1, "role": "assistant", "text": "Of course."},
    ]

    assert reviewer(transcript) == assessment
    call = responses.parse_calls[0]
    assert call["model"] == "assistant-model"
    assert call["instructions"] == REVIEWER_SYSTEM_PROMPT
    assert json.loads(call["input"][0]["content"]) == transcript
    assert call["input"][0]["role"] == "user"
    assert call["text_format"] is DialogueAssessment
    assert call["store"] is False
    assert call["max_output_tokens"] == 456


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(
            status="incomplete",
            output_parsed=DialogueAssessment(decision="continue", reason="Partial review."),
        ),
        SimpleNamespace(status="completed", output_parsed=None),
        SimpleNamespace(
            status="completed",
            output_parsed={"decision": "continue", "reason": "Not parsed by the SDK."},
        ),
    ],
)
def test_reviewer_rejects_incomplete_refused_or_invalid_responses(response):
    reviewer = create_dialogue_reviewer(
        FakeClient(FakeResponses(parsed_response=response)),
        "assistant-model",
        2048,
    )

    with pytest.raises(GenerationError):
        reviewer([{"turn_index": 0, "role": "user", "text": "Hello"}])
