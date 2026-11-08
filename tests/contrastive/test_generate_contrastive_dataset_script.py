import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_generation.contrastive.catalog import CATALOG
from agentic_generation.contrastive.generation import ContrastiveBatch
from agentic_generation.contrastive.models import ContrastCondition, ContrastivePair
from agentic_generation.contrastive.response_generation import ResponseGenerationResult

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "generate_contrastive_dataset.py"
SCRIPT_SPEC = spec_from_file_location("generate_contrastive_dataset", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
script = module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(script)


def make_pair(*, reverse_conditions=False):
    conditions = [
        ContrastCondition(
            appraisal="wanted",
            private_context="Friday is the speaker's preferred day.",
            vocal_direction="Sound pleased and relaxed.",
            response_goals=[
                "Acknowledge the welcome outcome.",
                "Show interest.",
            ],
        ),
        ContrastCondition(
            appraisal="unwanted",
            private_context="Friday conflicts with the speaker's plans.",
            vocal_direction="Sound restrained and displeased.",
            response_goals=[
                "Acknowledge the unwelcome outcome.",
                "Invite clarification.",
            ],
        ),
    ]
    if reverse_conditions:
        conditions.reverse()
    return ContrastivePair(
        utterance="They scheduled the visit for Friday.",
        appraisal_axis="wanted_vs_unwanted",
        conditions=conditions,
    )


def test_dry_run_describes_all_axis_files_and_calls(tmp_path, capsys):
    assert (
        script.main(
            [
                "--count-per-axis",
                "50",
                "--output-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["total_target_pairs"] == 400
    assert output["minimum_candidate_calls"] == 72
    assert output["response_calls"] == 800
    assert len(output["output_files"]) == len(CATALOG)
    assert str(tmp_path / "wanted_vs_unwanted" / "dataset.json") in output["output_files"]


@pytest.mark.parametrize(
    ("selected_goal_index", "selected_response_goal"),
    [(2, "Acknowledge the welcome outcome."), (1, "Acknowledge the welcome outcome.")],
)
def test_dataset_condition_rejects_inconsistent_goal_selection(
    selected_goal_index,
    selected_response_goal,
):
    with pytest.raises(ValueError, match="selected_.*goal"):
        script.DatasetCondition(
            private_context="Friday is preferred.",
            vocal_direction="Sound pleased.",
            response_goals=["Acknowledge the welcome outcome.", "Show interest."],
            selected_goal_index=selected_goal_index,
            selected_response_goal=selected_response_goal,
            response=script.DatasetResponse(
                text="That worked out well.",
                vocal_direction="Warm and concise.",
            ),
        )


@pytest.mark.parametrize("reverse_conditions", [False, True])
def test_generate_axis_passes_and_stores_first_goal_by_appraisal(
    tmp_path,
    monkeypatch,
    reverse_conditions,
):
    generation_calls = []
    response_calls = []
    candidate = make_pair(reverse_conditions=reverse_conditions)
    original_candidate = candidate.model_dump()

    def fake_generate_pairs(_client, request, _model, _max_output_tokens):
        generation_calls.append(request)
        return ContrastiveBatch(pairs=[candidate])

    def fake_generate_response(_client, request, _model, _max_output_tokens):
        response_calls.append(request)
        return ResponseGenerationResult(
            response_text=f"Response for {request.appraisal}.",
            response_vocal_direction=f"Respond in a {request.appraisal} manner.",
        )

    monkeypatch.setattr(script, "generate_contrastive_pairs", fake_generate_pairs)
    monkeypatch.setattr(script, "generate_response", fake_generate_response)
    client = SimpleNamespace()

    dataset = script.generate_axis(
        client,
        contrast_id="wanted_vs_unwanted",
        target_count=1,
        batch_size=1,
        max_batches=2,
        output_dir=tmp_path,
        model="test-model",
        max_output_tokens=512,
    )

    assert dataset.status == "completed"
    assert dataset.response_goal_policy == script.RESPONSE_GOAL_POLICY
    assert len(generation_calls) == 1
    assert len(response_calls) == 2
    requests_by_appraisal = {request.appraisal: request for request in response_calls}
    assert set(requests_by_appraisal) == {"wanted", "unwanted"}
    assert requests_by_appraisal["wanted"].model_dump(exclude_none=True) == {
        "utterance": "They scheduled the visit for Friday.",
        "appraisal": "wanted",
        "response_goal": "Acknowledge the welcome outcome.",
    }
    assert requests_by_appraisal["unwanted"].model_dump(exclude_none=True) == {
        "utterance": "They scheduled the visit for Friday.",
        "appraisal": "unwanted",
        "response_goal": "Acknowledge the unwelcome outcome.",
    }
    assert candidate.model_dump() == original_candidate

    output_path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["response_goal_policy"] == script.RESPONSE_GOAL_POLICY
    pair = output["pairs"][0]
    assert pair["utterance"] == "They scheduled the visit for Friday."
    assert set(pair["conditions"]) == {"wanted", "unwanted"}
    wanted = pair["conditions"]["wanted"]
    assert wanted["private_context"]
    assert wanted["vocal_direction"]
    assert wanted["response_goals"] == [
        "Acknowledge the welcome outcome.",
        "Show interest.",
    ]
    assert wanted["selected_goal_index"] == 0
    assert wanted["selected_response_goal"] == wanted["response_goals"][0]
    assert wanted["selected_response_goal"] == requests_by_appraisal["wanted"].response_goal
    assert wanted["response"] == {
        "text": "Response for wanted.",
        "vocal_direction": "Respond in a wanted manner.",
    }
    unwanted = pair["conditions"]["unwanted"]
    assert unwanted["response_goals"] == [
        "Acknowledge the unwelcome outcome.",
        "Invite clarification.",
    ]
    assert unwanted["selected_goal_index"] == 0
    assert unwanted["selected_response_goal"] == unwanted["response_goals"][0]
    assert unwanted["selected_response_goal"] == requests_by_appraisal["unwanted"].response_goal

    resumed = script.generate_axis(
        client,
        contrast_id="wanted_vs_unwanted",
        target_count=1,
        batch_size=1,
        max_batches=2,
        output_dir=tmp_path,
        model="test-model",
        max_output_tokens=512,
    )
    assert resumed.completed_pair_count == 1
    assert len(generation_calls) == 1
    assert len(response_calls) == 2


@pytest.mark.parametrize("stored_policy", [None, "another_policy"])
def test_resume_rejects_missing_or_different_goal_policy(tmp_path, stored_policy):
    path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    path.parent.mkdir(parents=True)
    stored = script._new_dataset("wanted_vs_unwanted", 1, "test-model").model_dump()
    if stored_policy is None:
        stored.pop("response_goal_policy")
    else:
        stored["response_goal_policy"] = stored_policy
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError, match="use a new output directory"):
        script.generate_axis(
            SimpleNamespace(),
            contrast_id="wanted_vs_unwanted",
            target_count=1,
            batch_size=1,
            max_batches=1,
            output_dir=tmp_path,
            model="test-model",
            max_output_tokens=512,
        )


def test_response_failure_does_not_checkpoint_partial_pair(tmp_path, monkeypatch):
    candidate = make_pair()
    original_candidate = candidate.model_dump()
    response_calls = []
    path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    script._write_dataset(
        path,
        script._new_dataset("wanted_vs_unwanted", 1, "test-model"),
    )

    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[candidate]),
    )

    def fail_second_response(_client, request, _model, _max_output_tokens):
        response_calls.append(request)
        if len(response_calls) == 2:
            raise RuntimeError("provider failed")
        return ResponseGenerationResult(
            response_text="First response.",
            response_vocal_direction="Calm.",
        )

    monkeypatch.setattr(script, "generate_response", fail_second_response)

    with pytest.raises(RuntimeError, match="provider failed"):
        script.generate_axis(
            SimpleNamespace(),
            contrast_id="wanted_vs_unwanted",
            target_count=1,
            batch_size=1,
            max_batches=1,
            output_dir=tmp_path,
            model="test-model",
            max_output_tokens=512,
        )

    assert len(response_calls) == 2
    assert candidate.model_dump() == original_candidate
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["completed_pair_count"] == 0
    assert stored["candidate_calls"] == 0
    assert stored["response_calls"] == 0
    assert stored["pairs"] == []
