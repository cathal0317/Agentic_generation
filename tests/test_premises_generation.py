import json
from types import SimpleNamespace

import pytest

from agentic_generation.openai_adapter import GenerationError
from agentic_generation.premises import Premise, PremiseBatch, propose_premises
from agentic_generation.spec import ConversationSpec


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
        raise AssertionError("text generation must not run")


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


@pytest.fixture
def spec():
    return ConversationSpec(
        scenario="Yesterday, she stumbled. She did not fall.",
        user_personality={"communication_style": "reserved"},
    )


@pytest.fixture
def batch():
    return PremiseBatch(
        premises=[
            Premise(
                concern="Still shaken by yesterday's stumble.",
                opening="I am still shaken by yesterday.",
                source_quotes=["Yesterday, she stumbled."],
            )
        ]
    )


def test_propose_premises_uses_one_structured_request(spec, batch):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=batch))

    result = propose_premises(FakeClient(responses), spec, "user-model", 512)

    assert result == batch
    assert len(responses.parse_calls) == 1
    call = responses.parse_calls[0]
    assert call["model"] == "user-model"
    assert call["text_format"] is PremiseBatch
    assert call["store"] is False
    assert call["max_output_tokens"] == 512
    assert json.loads(call["input"][0]["content"]) == spec.model_dump()
    assert responses.create_calls == []


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_propose_premises_rejects_incomplete_responses(spec, batch, status):
    responses = FakeResponses(SimpleNamespace(status=status, output_parsed=batch))

    with pytest.raises(GenerationError, match="No complete"):
        propose_premises(FakeClient(responses), spec, "model", 512)


def test_propose_premises_rejects_refusal(spec):
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=None))

    with pytest.raises(GenerationError, match="No complete"):
        propose_premises(FakeClient(responses), spec, "model", 512)


def test_propose_premises_rejects_quote_missing_from_scenario(spec, batch):
    batch.premises[0].source_quotes = ["This morning, she fell."]
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=batch))

    with pytest.raises(GenerationError, match="quotation"):
        propose_premises(FakeClient(responses), spec, "model", 512)


def test_propose_premises_rejects_normalized_duplicate_openings(spec, batch):
    batch.premises.append(
        batch.premises[0].model_copy(update={"opening": "  I AM still shaken   by yesterday.  "})
    )
    responses = FakeResponses(SimpleNamespace(status="completed", output_parsed=batch))

    with pytest.raises(GenerationError, match="repeated openings"):
        propose_premises(FakeClient(responses), spec, "model", 512)


def test_propose_premises_propagates_provider_errors(spec):
    responses = FakeResponses(error=RuntimeError("provider failed"))

    with pytest.raises(RuntimeError, match="provider failed"):
        propose_premises(FakeClient(responses), spec, "model", 512)
