import json

import pytest
from pydantic import ValidationError

from agentic_generation.premises import (
    PremiseSelectionError,
    ScenarioContract,
    load_premise_selection,
)
from agentic_generation.spec import ConversationSpec


@pytest.fixture
def spec():
    return ConversationSpec(
        scenario="Yesterday, she stumbled. She did not fall.",
        user_personality={"communication_style": "reserved"},
        max_exchanges=8,
    )


@pytest.fixture
def artifact_data(spec):
    return {
        "spec": spec.model_dump(),
        "model": "user-model",
        "max_output_tokens": 2048,
        "prompt_version": "premises-v1",
        "review_status": "unreviewed",
        "premises": [
            {
                "id": "p1",
                "concern": "Still shaken by yesterday's stumble.",
                "opening": "I am still shaken by yesterday.",
                "source_quotes": ["Yesterday, she stumbled."],
            },
            {
                "id": "p4",
                "concern": "Worried about going outside again.",
                "opening": "I am not sure I want to go out today.",
                "source_quotes": ["She did not fall."],
            },
        ],
    }


def write_artifact(tmp_path, data):
    path = tmp_path / "premises.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def contract_data():
    return {
        "focus": "Fear after the stumble.",
        "private_goal": "Talk through the fear before deciding what to do.",
        "tension": {
            "wants": "To keep walking independently.",
            "worries_about": "Stumbling again while alone.",
        },
        "disclosure_policy": ["Mention the stumble when asked."],
        "change_conditions": ["Consider suggestions that address the fear."],
        "source_quotes": ["Yesterday, she stumbled."],
    }


def test_accepts_valid_grounded_contract(tmp_path, spec, artifact_data):
    artifact_data["premises"][0]["contract"] = contract_data()
    path = write_artifact(tmp_path, artifact_data)

    selection = load_premise_selection(path, "p1", spec)

    assert selection.premise.contract is not None
    assert selection.premise.contract.focus == "Fear after the stumble."


@pytest.mark.parametrize(
    "invalid_contract",
    [
        {**contract_data(), "focus": "   "},
        {**contract_data(), "tension": {"wants": "", "worries_about": "A repeat."}},
        {**contract_data(), "disclosure_policy": [" "]},
        {**contract_data(), "disclosure_policy": []},
        {**contract_data(), "change_conditions": []},
        {**contract_data(), "source_quotes": []},
        {**contract_data(), "unknown": "not allowed"},
    ],
)
def test_rejects_invalid_contract_shape(invalid_contract):
    with pytest.raises(ValidationError):
        ScenarioContract.model_validate(invalid_contract)


def test_rejects_ungrounded_contract_quote(tmp_path, spec, artifact_data):
    contract = contract_data()
    contract["source_quotes"] = ["She fell this morning."]
    artifact_data["premises"][0]["contract"] = contract
    path = write_artifact(tmp_path, artifact_data)

    with pytest.raises(PremiseSelectionError, match="contract supporting quotation"):
        load_premise_selection(path, "p1", spec)


def test_loads_selected_premise_with_generation_metadata(tmp_path, spec, artifact_data):
    path = write_artifact(tmp_path, artifact_data)

    selection = load_premise_selection(path, "p4", spec)

    assert selection.premise.id == "p4"
    assert selection.premise.opening == "I am not sure I want to go out today."
    assert selection.prompt_version == "premises-v1"
    assert selection.model == "user-model"
    assert selection.premise.contract is None


def test_rejects_artifact_generated_from_different_spec(tmp_path, spec, artifact_data):
    artifact_data["spec"]["user_personality"] = {"communication_style": "outgoing"}
    path = write_artifact(tmp_path, artifact_data)

    with pytest.raises(PremiseSelectionError, match="different scenario"):
        load_premise_selection(path, "p1", spec)


def test_rejects_unknown_premise_id(tmp_path, spec, artifact_data):
    path = write_artifact(tmp_path, artifact_data)

    with pytest.raises(PremiseSelectionError, match="not found"):
        load_premise_selection(path, "p5", spec)


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"not": "a premise artifact"},
        {
            "spec": {"scenario": "A scenario."},
            "model": "model",
            "max_output_tokens": 2048,
            "prompt_version": "premises-v1",
            "review_status": "unreviewed",
            "premises": [],
        },
    ],
)
def test_rejects_malformed_artifact(tmp_path, spec, invalid_data):
    path = write_artifact(tmp_path, invalid_data)

    with pytest.raises(PremiseSelectionError, match="Invalid premise artifact"):
        load_premise_selection(path, "p1", spec)
