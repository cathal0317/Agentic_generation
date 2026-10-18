import json
import sys
from types import SimpleNamespace

import pytest

import agentic_generation.cli as cli
from agentic_generation.controller import DialogueAssessment
from agentic_generation.conversation import OPENING_USER_REQUEST
from agentic_generation.openai_adapter import GenerationError


class FakeResponses:
    def __init__(self, outputs=None, error=None, assessments=None, review_error=None):
        self.outputs = iter(outputs or [])
        self.error = error
        self.assessments = iter(assessments or [])
        self.review_error = review_error
        self.calls = []
        self.parse_calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(status="completed", output_text=next(self.outputs))

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        if self.review_error is not None:
            raise self.review_error
        return SimpleNamespace(status="completed", output_parsed=next(self.assessments))


class FakeClient:
    def __init__(self, outputs=None, error=None, assessments=None, review_error=None):
        self.responses = FakeResponses(outputs, error, assessments, review_error)
        self.closed = False

    def close(self):
        self.closed = True


def write_scenario(tmp_path, max_exchanges=1):
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps(
            {
                "scenario": "A test scenario.",
                "user_personality": {"style": "reserved"},
                "max_exchanges": max_exchanges,
            }
        ),
        encoding="utf-8",
    )
    return path


def write_premises(tmp_path, scenario_path, contract=None):
    premise = {
        "id": "p1",
        "concern": "PRIVATE_SELECTED_CONCERN",
        "opening": "The situation yesterday is still bothering me.",
        "source_quotes": ["A test scenario."],
    }
    if contract is not None:
        premise["contract"] = contract

    path = tmp_path / "premises.json"
    path.write_text(
        json.dumps(
            {
                "spec": json.loads(scenario_path.read_text()),
                "model": "candidate-model",
                "max_output_tokens": 512,
                "prompt_version": "premises-v1",
                "review_status": "unreviewed",
                "premises": [premise],
            }
        ),
        encoding="utf-8",
    )
    return path


def configure_generation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    monkeypatch.setenv("OPENAI_USER_MODEL", "user-model")
    monkeypatch.setenv("OPENAI_ASSISTANT_MODEL", "assistant-model")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "456")


def run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["agentic-generation", *arguments])
    return cli.main()


def test_generate_command_uses_real_orchestrator_and_prints_json(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)
    configure_generation(monkeypatch)
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        ["Hello from the user.", "Hello from the assistant."],
        assessments=[
            DialogueAssessment(
                decision="end",
                reason="The exchange reached a natural closing.",
            )
        ],
    )
    constructor_calls = []

    def fake_openai(**kwargs):
        constructor_calls.append(kwargs)
        return client

    monkeypatch.setattr(cli, "OpenAI", fake_openai)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 0
    output = json.loads(capsys.readouterr().out)

    assert constructor_calls == [{"api_key": "test-secret-key", "timeout": 60.0, "max_retries": 0}]
    assert output["spec"]["scenario"] == "A test scenario."
    assert output["models"] == {
        "user": "user-model",
        "assistant": "assistant-model",
        "reviewer": "assistant-model",
    }
    assert output["generation_settings"] == {
        "max_output_tokens": 456,
        "timeout_seconds": 60.0,
        "max_retries": 0,
    }
    assert output["turns"] == [
        {"turn_index": 0, "role": "user", "text": "Hello from the user."},
        {
            "turn_index": 1,
            "role": "assistant",
            "text": "Hello from the assistant.",
        },
    ]
    assert [call["model"] for call in client.responses.calls] == [
        "user-model",
        "assistant-model",
    ]
    assert client.responses.calls[0]["input"] == [{"role": "user", "content": OPENING_USER_REQUEST}]
    assert client.responses.calls[1]["input"] == [
        {"role": "user", "content": "Hello from the user."}
    ]
    assert all(call["max_output_tokens"] == 456 for call in client.responses.calls)
    assert output["termination_reason"] == "natural_end"
    assert output["control_trace"] == [
        {
            "after_turn_index": 1,
            "decision": "end",
            "reason": "The exchange reached a natural closing.",
        }
    ]
    assert len(client.responses.parse_calls) == 1
    assert client.closed


def test_generate_command_runs_selected_premise_and_records_provenance(
    tmp_path, monkeypatch, capsys
):
    scenario_path = write_scenario(tmp_path, max_exchanges=2)
    premise_path = write_premises(tmp_path, scenario_path)
    configure_generation(monkeypatch)
    client = FakeClient(
        [
            "What about it is still on your mind?",
            "I keep thinking about what could have happened.",
            "That uncertainty can linger.",
        ],
        assessments=[
            DialogueAssessment(decision="continue", reason="The dialogue is progressing."),
            DialogueAssessment(decision="end", reason="The concern has been explored."),
        ],
    )
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: client)

    assert (
        run_cli(
            monkeypatch,
            "generate",
            str(scenario_path),
            "--premises",
            str(premise_path),
            "--premise-id",
            "p1",
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["turns"][0]["text"] == ("The situation yesterday is still bothering me.")
    assert output["selected_premise"] == {
        "id": "p1",
        "concern": "PRIVATE_SELECTED_CONCERN",
        "opening": "The situation yesterday is still bothering me.",
        "source_quotes": ["A test scenario."],
        "contract": None,
        "generation": {
            "prompt_version": "premises-v1",
            "model": "candidate-model",
            "contract_version": None,
        },
    }
    assert [call["model"] for call in client.responses.calls] == [
        "assistant-model",
        "user-model",
        "assistant-model",
    ]
    assert "PRIVATE_SELECTED_CONCERN" in client.responses.calls[1]["instructions"]
    assert "PRIVATE_SELECTED_CONCERN" not in repr(
        [client.responses.calls[0], client.responses.calls[2]]
    )
    assert "PRIVATE_SELECTED_CONCERN" not in repr(client.responses.parse_calls)


def test_generate_command_passes_and_records_private_contract(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path, max_exchanges=2)
    contract = {
        "focus": "PRIVATE_FOCUS",
        "private_goal": "PRIVATE_GOAL",
        "tension": {
            "wants": "PRIVATE_WANT",
            "worries_about": "PRIVATE_WORRY",
        },
        "disclosure_policy": ["PRIVATE_DISCLOSURE"],
        "change_conditions": ["PRIVATE_CHANGE"],
        "source_quotes": ["A test scenario."],
    }
    premise_path = write_premises(tmp_path, scenario_path, contract)
    configure_generation(monkeypatch)
    client = FakeClient(
        [
            "A public assistant response.",
            "A later user response.",
            "A final assistant response.",
        ],
        assessments=[
            DialogueAssessment(decision="continue", reason="The dialogue continues."),
            DialogueAssessment(decision="end", reason="The dialogue is complete."),
        ],
    )
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: client)

    assert (
        run_cli(
            monkeypatch,
            "generate",
            str(scenario_path),
            "--premises",
            str(premise_path),
            "--premise-id",
            "p1",
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)

    assert output["selected_premise"]["contract"] == contract
    assert output["selected_premise"]["generation"]["contract_version"] == ("scenario-contract-v1")
    assert output["turns"][0]["text"] == ("The situation yesterday is still bothering me.")
    assert "PRIVATE_FOCUS" in client.responses.calls[1]["instructions"]
    assert "PRIVATE_FOCUS" not in repr([client.responses.calls[0], client.responses.calls[2]])
    assert "PRIVATE_FOCUS" not in repr(client.responses.parse_calls)


def test_generate_rejects_ungrounded_contract_before_provider_call(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)
    contract = {
        "focus": "A private focus.",
        "private_goal": "A private goal.",
        "tension": {"wants": "Independence.", "worries_about": "Falling."},
        "disclosure_policy": ["Disclose details gradually."],
        "change_conditions": ["Consider relevant suggestions."],
        "source_quotes": ["This quotation is absent."],
    }
    premise_path = write_premises(tmp_path, scenario_path, contract)

    def unexpected_client(**kwargs):
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(cli, "OpenAI", unexpected_client)

    assert (
        run_cli(
            monkeypatch,
            "generate",
            str(scenario_path),
            "--premises",
            str(premise_path),
            "--premise-id",
            "p1",
        )
        == 2
    )
    assert "contract supporting quotation" in capsys.readouterr().err


def test_generate_command_requires_both_premise_options(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)

    assert (
        run_cli(
            monkeypatch,
            "generate",
            str(scenario_path),
            "--premise-id",
            "p1",
        )
        == 2
    )
    assert "must be provided together" in capsys.readouterr().err


def test_generate_command_rejects_missing_configuration(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)
    monkeypatch.chdir(tmp_path)
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_USER_MODEL",
        "OPENAI_ASSISTANT_MODEL",
        "OPENAI_MAX_OUTPUT_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)

    def unexpected_client(**kwargs):
        raise AssertionError("client must not be created")

    monkeypatch.setattr(cli, "OpenAI", unexpected_client)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Invalid OpenAI settings" in captured.err


def test_generate_command_sanitizes_provider_errors_and_closes_client(
    tmp_path, monkeypatch, capsys
):
    scenario_path = write_scenario(tmp_path)
    configure_generation(monkeypatch)
    monkeypatch.chdir(tmp_path)
    client = FakeClient(error=RuntimeError("raw provider body with test-secret-key"))
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: client)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Conversation generation failed" in captured.err
    assert "raw provider body" not in captured.err
    assert "test-secret-key" not in captured.err
    assert client.closed


def test_generate_command_reports_safe_generation_stage(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)
    configure_generation(monkeypatch)
    client = FakeClient()
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: client)

    def incomplete_generation(*args, **kwargs):
        raise GenerationError("user response did not complete: max_output_tokens")

    monkeypatch.setattr(cli, "generate_conversation", incomplete_generation)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        "Conversation generation failed: user response did not complete: max_output_tokens"
    ) in captured.err
    assert client.closed


@pytest.mark.parametrize(
    ("error_name", "expected_message"),
    [
        ("APIConnectionError", "could not connect to OpenAI"),
        ("APITimeoutError", "OpenAI request timed out"),
        ("APIStatusError", "OpenAI returned HTTP 429"),
    ],
)
def test_generate_command_classifies_provider_errors_safely(
    error_name,
    expected_message,
    tmp_path,
    monkeypatch,
    capsys,
):
    class SafeProviderError(Exception):
        status_code = 429

    scenario_path = write_scenario(tmp_path)
    configure_generation(monkeypatch)
    client = FakeClient()
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(cli, error_name, SafeProviderError)

    def failed_generation(*args, **kwargs):
        raise SafeProviderError("raw provider body with test-secret-key")

    monkeypatch.setattr(cli, "generate_conversation", failed_generation)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 1
    captured = capsys.readouterr()
    assert expected_message in captured.err
    assert "raw provider body" not in captured.err
    assert "test-secret-key" not in captured.err
    assert client.closed


def test_review_errors_use_the_safe_generation_failure_path(tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)
    configure_generation(monkeypatch)
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        ["Hello from the user.", "Hello from the assistant."],
        review_error=RuntimeError("raw review response"),
    )
    monkeypatch.setattr(cli, "OpenAI", lambda **kwargs: client)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Conversation generation failed" in captured.err
    assert "raw review response" not in captured.err
    assert client.closed


@pytest.mark.parametrize("command", ["validate-scenario", "preview-prompts"])
def test_scenario_commands_do_not_load_openai_configuration(command, tmp_path, monkeypatch, capsys):
    scenario_path = write_scenario(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def unexpected_settings_load():
        raise AssertionError("OpenAI settings must not be loaded")

    monkeypatch.setattr(cli, "GenerationSettings", unexpected_settings_load)

    assert run_cli(monkeypatch, command, str(scenario_path)) == 0
    assert capsys.readouterr().err == ""
