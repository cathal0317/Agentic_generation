import json
import sys

import pytest
from pydantic import ValidationError

import agentic_generation.cli as cli
from agentic_generation.spec import ConversationSpec


def test_valid_spec_trims_scenario_and_applies_defaults():
    spec = ConversationSpec.model_validate({"scenario": "  A difficult conversation.  "})

    assert spec.scenario == "A difficult conversation."
    assert spec.user_personality == {}
    assert spec.max_exchanges == 8


def test_valid_spec_accepts_personality_and_exchange_count():
    spec = ConversationSpec.model_validate(
        {
            "scenario": "A meeting was postponed.",
            "user_personality": {"communication_style": "reserved"},
            "max_exchanges": 2,
        }
    )

    assert spec.user_personality == {"communication_style": "reserved"}
    assert spec.max_exchanges == 2


@pytest.mark.parametrize("scenario", [None, "", " \n\t"])
def test_spec_rejects_missing_or_blank_scenario(scenario):
    data = {} if scenario is None else {"scenario": scenario}

    with pytest.raises(ValidationError):
        ConversationSpec.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {"scenario": 123},
        {"scenario": "Valid", "user_personality": []},
        {"scenario": "Valid", "max_exchanges": "4"},
        {"scenario": "Valid", "max_exchanges": 4.0},
        {"scenario": "Valid", "max_exchanges": True},
    ],
)
def test_spec_rejects_incorrect_field_types(data):
    with pytest.raises(ValidationError):
        ConversationSpec.model_validate(data)


def test_spec_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ConversationSpec.model_validate({"scenario": "Valid", "initial_context": "Not allowed"})


@pytest.mark.parametrize("max_exchanges", [0, -1])
def test_spec_rejects_non_positive_exchange_counts(max_exchanges):
    with pytest.raises(ValidationError):
        ConversationSpec.model_validate({"scenario": "Valid", "max_exchanges": max_exchanges})


def run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["agentic-generation", *arguments])
    return cli.main()


def test_cli_validates_without_database_configuration(tmp_path, monkeypatch, capsys):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps({"scenario": "A valid scenario."}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    def unexpected_settings_load():
        raise AssertionError("database settings must not be loaded")

    monkeypatch.setattr(cli, "Settings", unexpected_settings_load)

    assert run_cli(monkeypatch, "validate-scenario", str(scenario_path)) == 0
    assert "Scenario is valid" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        ("{not json", "Invalid JSON"),
        (json.dumps({"scenario": "   "}), "Invalid scenario input"),
    ],
)
def test_cli_reports_invalid_files_without_traceback(
    tmp_path, monkeypatch, capsys, contents, expected_error
):
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(contents, encoding="utf-8")

    assert run_cli(monkeypatch, "validate-scenario", str(scenario_path)) == 2
    error = capsys.readouterr().err
    assert expected_error in error
    assert "Traceback" not in error


def test_cli_reports_unreadable_file(tmp_path, monkeypatch, capsys):
    missing_path = tmp_path / "missing.json"

    assert run_cli(monkeypatch, "validate-scenario", str(missing_path)) == 2
    error = capsys.readouterr().err
    assert "Could not read scenario file" in error
    assert "Traceback" not in error
