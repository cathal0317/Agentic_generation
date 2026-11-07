import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
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

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "compare_response_goals.py"
SCRIPT_SPEC = spec_from_file_location("compare_response_goals", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
comparison = module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(comparison)


class FakeResponses:
    def __init__(self, fail_on_call=None):
        self.fail_on_call = fail_on_call
        self.parse_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if len(self.parse_calls) == self.fail_on_call:
            raise RuntimeError("provider failed")
        return SimpleNamespace(
            status="completed",
            output_parsed=ResponseGenerationResult(
                response_text=f"Reply {len(self.parse_calls)}.",
                response_vocal_direction="Calm and attentive.",
            ),
        )


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


@pytest.fixture
def input_data():
    return comparison.ComparisonInput(
        cases=[
            comparison.ComparisonCase(
                id="meeting_wanted",
                utterance="They moved the meeting to Friday.",
                appraisal="wanted",
                original_goal=(
                    "Briefly acknowledge their positive stance about the meeting being moved."
                ),
                concise_goal="Share their positive reaction to the change.",
            )
        ]
    )


def test_one_case_makes_three_isolated_calls_with_common_settings(input_data):
    responses = FakeResponses()

    output = comparison.run_comparison(
        FakeClient(responses),
        input_data,
        "test-model",
        512,
    )

    assert output["status"] == "completed"
    assert output["requested_calls"] == output["completed_calls"] == 3
    assert [result["variant"] for result in output["results"]] == [
        "original_goal",
        "concise_goal",
        "no_goal",
    ]
    assert comparison.generate_response is generate_response
    assert all(
        isinstance(request, ResponseGenerationRequest)
        for _, request in comparison.build_variant_requests(input_data.cases[0])
    )
    assert len(responses.parse_calls) == 3
    payloads = [json.loads(call["input"][0]["content"]) for call in responses.parse_calls]
    assert payloads == [
        {
            "utterance": "They moved the meeting to Friday.",
            "appraisal": "wanted",
            "response_goal": input_data.cases[0].original_goal,
        },
        {
            "utterance": "They moved the meeting to Friday.",
            "appraisal": "wanted",
            "response_goal": input_data.cases[0].concise_goal,
        },
        {
            "utterance": "They moved the meeting to Friday.",
            "appraisal": "wanted",
        },
    ]
    serialized_payloads = json.dumps(payloads)
    assert "Reply 1" not in serialized_payloads
    assert "Reply 2" not in serialized_payloads
    for call in responses.parse_calls:
        assert call["instructions"] == SYSTEM_PROMPT
        assert call["model"] == "test-model"
        assert call["max_output_tokens"] == 512
        assert call["store"] is False
        assert call["text_format"] is ResponseGenerationResult
    assert output["prompt_version"] == RESPONSE_GENERATION_PROMPT_VERSION
    assert output["prompt_hash"] == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def test_v2_prompt_contains_requested_safety_additions():
    assert comparison.RESPONSE_GENERATION_PROMPT_VERSION == "response-generation-v2"
    assert comparison.SYSTEM_PROMPT == SYSTEM_PROMPT
    assert "Do not invent the cause of the appraisal" in comparison.SYSTEM_PROMPT
    assert "motives, character,\nor likely performance" in comparison.SYSTEM_PROMPT
    assert "Acknowledgment does not require reassurance, advice, or a positive reframing." in (
        comparison.SYSTEM_PROMPT
    )


@pytest.mark.parametrize(
    "data",
    [
        {"cases": []},
        {
            "cases": [
                {
                    "id": "duplicate",
                    "utterance": "An event happened.",
                    "appraisal": "positive",
                    "original_goal": "Acknowledge it.",
                    "concise_goal": "Acknowledge it.",
                },
                {
                    "id": "duplicate",
                    "utterance": "Another event happened.",
                    "appraisal": "negative",
                    "original_goal": "Acknowledge it.",
                    "concise_goal": "Acknowledge it.",
                },
            ]
        },
        {
            "cases": [
                {
                    "id": "blank",
                    "utterance": " ",
                    "appraisal": "positive",
                    "original_goal": "Acknowledge it.",
                    "concise_goal": "Acknowledge it.",
                }
            ]
        },
        {
            "cases": [
                {
                    "id": "extra",
                    "utterance": "An event happened.",
                    "appraisal": "positive",
                    "original_goal": "Acknowledge it.",
                    "concise_goal": "Acknowledge it.",
                    "private_context": "not allowed",
                }
            ]
        },
    ],
)
def test_invalid_input_is_rejected(data):
    with pytest.raises(ValidationError):
        comparison.ComparisonInput.model_validate(data)


def test_dry_run_validates_and_does_not_create_client(
    tmp_path,
    monkeypatch,
    capsys,
    input_data,
):
    path = tmp_path / "input.json"
    path.write_text(input_data.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        comparison,
        "OpenAI",
        lambda **kwargs: pytest.fail("dry-run must not create a client"),
    )
    monkeypatch.setattr(
        comparison,
        "GenerationSettings",
        lambda: pytest.fail("dry-run must not load settings"),
    )

    assert comparison.main([str(path), "--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "dry_run"
    assert output["expected_calls"] == 3
    assert "response_goal" not in output["calls"][2]["request"]


def test_invalid_file_is_rejected_before_client(tmp_path, monkeypatch, capsys):
    path = tmp_path / "invalid.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        comparison,
        "OpenAI",
        lambda **kwargs: pytest.fail("invalid input must not create a client"),
    )
    monkeypatch.setattr(
        comparison,
        "GenerationSettings",
        lambda: pytest.fail("invalid input must not load settings"),
    )

    assert comparison.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid response-goal comparison input" in captured.err


def test_midstream_failure_stops_and_preserves_completed_results(input_data):
    responses = FakeResponses(fail_on_call=2)

    output = comparison.run_comparison(
        FakeClient(responses),
        input_data,
        "model",
        256,
    )

    assert output["status"] == "failed"
    assert output["completed_calls"] == 1
    assert len(output["results"]) == 1
    assert len(responses.parse_calls) == 2
    assert output["failure"] == {
        "case_id": "meeting_wanted",
        "variant": "concise_goal",
        "reason": "unexpected_error",
    }
    assert output["prompt_version"] == RESPONSE_GENERATION_PROMPT_VERSION
    assert output["prompt_hash"] == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
