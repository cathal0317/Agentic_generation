import json
import sys
from types import SimpleNamespace

import pytest

import agentic_generation.cli as cli
from agentic_generation.conversation import OPENING_USER_REQUEST


class FakeResponses:
    def __init__(self, outputs=None, error=None):
        self.outputs = iter(outputs or [])
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(status="completed", output_text=next(self.outputs))


class FakeClient:
    def __init__(self, outputs=None, error=None):
        self.responses = FakeResponses(outputs, error)
        self.closed = False

    def close(self):
        self.closed = True


def write_scenario(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps(
            {
                "scenario": "A test scenario.",
                "user_personality": {"style": "reserved"},
                "max_exchanges": 1,
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
    client = FakeClient(["Hello from the user.", "Hello from the assistant."])
    constructor_calls = []

    def fake_openai(**kwargs):
        constructor_calls.append(kwargs)
        return client

    monkeypatch.setattr(cli, "OpenAI", fake_openai)

    assert run_cli(monkeypatch, "generate", str(scenario_path)) == 0
    output = json.loads(capsys.readouterr().out)

    assert constructor_calls == [{"api_key": "test-secret-key", "timeout": 60.0, "max_retries": 0}]
    assert output["spec"]["scenario"] == "A test scenario."
    assert output["models"] == {"user": "user-model", "assistant": "assistant-model"}
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
    assert client.closed


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
