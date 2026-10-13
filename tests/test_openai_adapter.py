from types import SimpleNamespace

import pytest

from agentic_generation.openai_adapter import GenerationError, create_text_generator


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


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
