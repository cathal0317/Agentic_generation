import json

import pytest

from agentic_generation.premises import (
    PremiseSelectionError,
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


def test_loads_selected_premise_with_generation_metadata(tmp_path, spec, artifact_data):
    path = write_artifact(tmp_path, artifact_data)

    selection = load_premise_selection(path, "p4", spec)

    assert selection.premise.id == "p4"
    assert selection.premise.opening == "I am not sure I want to go out today."
    assert selection.prompt_version == "premises-v1"
    assert selection.model == "user-model"


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
