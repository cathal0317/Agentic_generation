import json
import sys

import agentic_generation.cli as cli
from agentic_generation.prompts import build_assistant_system_prompt, build_user_system_prompt
from agentic_generation.spec import ConversationSpec


def test_user_prompt_contains_scenario_and_personality():
    spec = ConversationSpec(
        scenario="The user is preparing for a difficult conversation.",
        user_personality={"communication_style": "reserved"},
    )

    prompt = build_user_system_prompt(spec)

    assert spec.scenario in prompt
    assert "communication_style" in prompt
    assert "reserved" in prompt


def test_scenario_changes_do_not_change_assistant_prompt():
    first = ConversationSpec(
        scenario="First private scenario.",
        user_personality={"private_detail": "first secret"},
    )
    second = ConversationSpec(
        scenario="Second private scenario.",
        user_personality={"private_detail": "second secret"},
    )

    assert build_user_system_prompt(first) != build_user_system_prompt(second)
    assert build_assistant_system_prompt() == build_assistant_system_prompt()
    assistant_prompt = build_assistant_system_prompt()
    assert "First private scenario." not in assistant_prompt
    assert "first secret" not in assistant_prompt
    assert "Second private scenario." not in assistant_prompt
    assert "second secret" not in assistant_prompt


def test_preview_command_outputs_json_without_database_configuration(tmp_path, monkeypatch, capsys):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "scenario": "A private scenario detail.",
                "user_personality": {"communication_style": "reserved"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    def unexpected_settings_load():
        raise AssertionError("database settings must not be loaded")

    monkeypatch.setattr(cli, "Settings", unexpected_settings_load)
    monkeypatch.setattr(
        sys,
        "argv",
        ["agentic-generation", "preview-prompts", str(scenario_path)],
    )

    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"user_system_prompt", "assistant_system_prompt"}
    assert "A private scenario detail." in output["user_system_prompt"]
    assert "A private scenario detail." not in output["assistant_system_prompt"]


def test_preview_command_preserves_validation_errors(tmp_path, monkeypatch, capsys):
    scenario_path = tmp_path / "invalid.json"
    scenario_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["agentic-generation", "preview-prompts", str(scenario_path)],
    )

    assert cli.main() == 2
    assert "Invalid JSON in scenario file" in capsys.readouterr().err
