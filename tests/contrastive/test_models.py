import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_generation.contrastive import ContrastivePair


def pair_data():
    return {
        "utterance": "They put me in charge.",
        "appraisal_axis": "opportunity_vs_burden",
        "conditions": [
            {
                "appraisal": "opportunity",
                "private_context": "The speaker wanted an opportunity to lead.",
                "vocal_direction": "Quietly proud, with restrained excitement.",
                "response_goals": [
                    "Acknowledge the opportunity.",
                    "Invite the speaker to share what they are looking forward to.",
                ],
            },
            {
                "appraisal": "burden",
                "private_context": (
                    "The speaker is already overworked and did not want the responsibility."
                ),
                "vocal_direction": "Tired and overwhelmed.",
                "response_goals": [
                    "Acknowledge the added pressure.",
                    "Ask whether support or flexibility is available.",
                ],
            },
        ],
    }


def test_example_loads_and_validates():
    path = Path("examples/contrastive/contrastive_pair.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    pair = ContrastivePair.model_validate(data)

    assert pair.utterance == "They put me in charge."
    assert [condition.appraisal for condition in pair.conditions] == ["opportunity", "burden"]


def test_strips_leading_and_trailing_whitespace():
    data = pair_data()
    data["utterance"] = "  They put me in charge. \n"
    data["appraisal_axis"] = "\t opportunity_vs_burden "
    data["conditions"][0]["appraisal"] = " opportunity "
    data["conditions"][0]["private_context"] = "  Wanted a chance to lead. "
    data["conditions"][0]["vocal_direction"] = " Quietly proud. "
    data["conditions"][0]["response_goals"] = [" Acknowledge the opportunity. "]

    pair = ContrastivePair.model_validate(data)

    assert pair.utterance == "They put me in charge."
    assert pair.appraisal_axis == "opportunity_vs_burden"
    assert pair.conditions[0].appraisal == "opportunity"
    assert pair.conditions[0].private_context == "Wanted a chance to lead."
    assert pair.conditions[0].vocal_direction == "Quietly proud."
    assert pair.conditions[0].response_goals == ["Acknowledge the opportunity."]


def test_rejects_blank_utterance():
    data = pair_data()
    data["utterance"] = " \n\t "

    with pytest.raises(ValidationError):
        ContrastivePair.model_validate(data)


@pytest.mark.parametrize("condition_count", [1, 3])
def test_rejects_condition_count_other_than_two(condition_count):
    data = pair_data()
    data["conditions"] = (data["conditions"] * 2)[:condition_count]

    with pytest.raises(ValidationError):
        ContrastivePair.model_validate(data)


def test_rejects_missing_required_condition_field():
    data = pair_data()
    del data["conditions"][0]["vocal_direction"]

    with pytest.raises(ValidationError):
        ContrastivePair.model_validate(data)


def test_rejects_empty_response_goals():
    data = pair_data()
    data["conditions"][0]["response_goals"] = []

    with pytest.raises(ValidationError):
        ContrastivePair.model_validate(data)


def test_rejects_duplicate_normalized_appraisals():
    data = pair_data()
    data["conditions"][0]["appraisal"] = "Opportunity"
    data["conditions"][1]["appraisal"] = " opportunity "

    with pytest.raises(ValidationError):
        ContrastivePair.model_validate(data)


@pytest.mark.parametrize("location", ["pair", "condition"])
def test_rejects_unknown_fields(location):
    data = pair_data()
    if location == "pair":
        data["unknown"] = "not allowed"
    else:
        data["conditions"][0]["unknown"] = "not allowed"

    with pytest.raises(ValidationError):
        ContrastivePair.model_validate(data)
