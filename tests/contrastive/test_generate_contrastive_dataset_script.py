import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_generation.contrastive.catalog import CATALOG
from agentic_generation.contrastive.generation import ContrastiveBatch
from agentic_generation.contrastive.models import ContrastCondition, ContrastivePair
from agentic_generation.contrastive.response_generation import ResponseGenerationResult
from agentic_generation.contrastive.response_goals import ResponseGoalsResult

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "generate_contrastive_dataset.py"
SCRIPT_SPEC = spec_from_file_location("generate_contrastive_dataset", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
script = module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(script)

PUBLIC_GOALS = {
    "wanted": [
        "Acknowledge that the event matches the speaker's current preference.",
        "Invite the speaker to say what they welcome about the event.",
    ],
    "unwanted": [
        "Acknowledge that the event conflicts with the speaker's current preference.",
        "Invite the speaker to say what concerns them about the event.",
    ],
}


@pytest.fixture(autouse=True)
def pass_candidate_event_review(monkeypatch):
    monkeypatch.setattr(
        script,
        "_has_consistent_shared_event",
        lambda *_args, **_kwargs: True,
    )


def make_pair(
    *,
    reverse_conditions=False,
    wanted_goal_count=2,
    unwanted_goal_count=2,
    utterance="They scheduled the visit for Friday.",
):
    wanted_goals = [
        "Acknowledge the welcome outcome.",
        "Show interest.",
        "Invite the speaker to share what this enables.",
    ]
    unwanted_goals = [
        "Acknowledge the unwelcome outcome.",
        "Invite clarification.",
        "Offer a brief supportive response.",
    ]
    conditions = [
        ContrastCondition(
            appraisal="wanted",
            private_context="Friday is the speaker's preferred day.",
            vocal_direction="Sound pleased and relaxed.",
            response_goals=wanted_goals[:wanted_goal_count],
        ),
        ContrastCondition(
            appraisal="unwanted",
            private_context="Friday conflicts with the speaker's plans.",
            vocal_direction="Sound restrained and displeased.",
            response_goals=unwanted_goals[:unwanted_goal_count],
        ),
    ]
    if reverse_conditions:
        conditions.reverse()
    return ContrastivePair(
        utterance=utterance,
        appraisal_axis="wanted_vs_unwanted",
        conditions=conditions,
    )


def install_fake_goal_generation(monkeypatch, calls=None):
    def fake_generate_response_goals(_client, request, _model, _max_output_tokens):
        if calls is not None:
            calls.append(request)
        return ResponseGoalsResult(response_goals=PUBLIC_GOALS[request.appraisal])

    monkeypatch.setattr(
        script,
        "generate_response_goals",
        fake_generate_response_goals,
    )


def test_first_dry_run_describes_all_axis_files_and_calls(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        script,
        "GenerationSettings",
        lambda: pytest.fail("dry-run must not load API settings"),
    )
    monkeypatch.setattr(
        script,
        "OpenAI",
        lambda **_kwargs: pytest.fail("dry-run must not create a client"),
    )
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
    assert output["goal_mode"] == "first"
    assert output["response_goal_policy"] == script.RESPONSE_GOAL_POLICY
    assert output["total_target_pairs"] == 400
    assert output["minimum_candidate_calls"] == 72
    assert output["minimum_candidate_review_calls"] == 400
    assert output["response_goal_calls"] == 800
    assert output["response_calls"] == 800
    assert len(output["output_files"]) == len(CATALOG)
    assert str(tmp_path / "wanted_vs_unwanted" / "dataset.json") in output["output_files"]


def test_all_dry_run_reports_dynamic_response_count_without_client(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setattr(
        script,
        "GenerationSettings",
        lambda: pytest.fail("dry-run must not load API settings"),
    )
    monkeypatch.setattr(
        script,
        "OpenAI",
        lambda **_kwargs: pytest.fail("dry-run must not create a client"),
    )

    assert (
        script.main(
            [
                "--contrast-id",
                "wanted_vs_unwanted",
                "--count-per-axis",
                "2",
                "--goal-mode",
                "all",
                "--output-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert output["goal_mode"] == "all"
    assert output["response_goal_policy"] == script.ALL_RESPONSE_GOALS_POLICY
    assert output["minimum_candidate_review_calls"] == 2
    assert output["response_goal_calls"] == 4
    assert output["response_calls"] == 8
    assert output["response_candidate_status"] == "unreviewed"


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


@pytest.mark.parametrize(
    "response_candidates",
    [
        [
            {
                "goal_index": 0,
                "response_goal": "Goal zero.",
                "response": {"text": "Zero.", "vocal_direction": "Calm."},
            }
        ],
        [
            {
                "goal_index": 0,
                "response_goal": "Goal zero.",
                "response": {"text": "Zero.", "vocal_direction": "Calm."},
            },
            {
                "goal_index": 0,
                "response_goal": "Goal zero.",
                "response": {"text": "Duplicate.", "vocal_direction": "Calm."},
            },
        ],
        [
            {
                "goal_index": 0,
                "response_goal": "Goal zero.",
                "response": {"text": "Zero.", "vocal_direction": "Calm."},
            },
            {
                "goal_index": 2,
                "response_goal": "Goal one.",
                "response": {"text": "Out of range.", "vocal_direction": "Calm."},
            },
        ],
        [
            {
                "goal_index": 0,
                "response_goal": "Wrong goal.",
                "response": {"text": "Mismatch.", "vocal_direction": "Calm."},
            },
            {
                "goal_index": 1,
                "response_goal": "Goal one.",
                "response": {"text": "One.", "vocal_direction": "Calm."},
            },
        ],
    ],
)
def test_all_goals_condition_rejects_invalid_candidate_mapping(response_candidates):
    with pytest.raises(ValueError, match="response candidate|response_candidates"):
        script.AllGoalsDatasetCondition(
            private_context="Private context.",
            vocal_direction="User direction.",
            response_goals=["Goal zero.", "Goal one."],
            response_candidates=response_candidates,
        )


@pytest.mark.parametrize("reverse_conditions", [False, True])
def test_generate_axis_passes_and_stores_first_goal_by_appraisal(
    tmp_path,
    monkeypatch,
    reverse_conditions,
):
    generation_calls = []
    goal_calls = []
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
    install_fake_goal_generation(monkeypatch, goal_calls)
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
    assert dataset.candidate_review_calls == 1
    assert dataset.response_goal_calls == 2
    assert len(generation_calls) == 1
    assert len(goal_calls) == 2
    assert len(response_calls) == 2
    assert {request.appraisal for request in goal_calls} == {"wanted", "unwanted"}
    assert all(
        request.model_dump()
        == {
            "utterance": "They scheduled the visit for Friday.",
            "appraisal": request.appraisal,
        }
        for request in goal_calls
    )
    assert "preferred day" not in repr(goal_calls)
    assert "conflicts with the speaker's plans" not in repr(goal_calls)
    requests_by_appraisal = {request.appraisal: request for request in response_calls}
    assert set(requests_by_appraisal) == {"wanted", "unwanted"}
    assert requests_by_appraisal["wanted"].model_dump(exclude_none=True) == {
        "utterance": "They scheduled the visit for Friday.",
        "appraisal": "wanted",
        "response_goal": PUBLIC_GOALS["wanted"][0],
    }
    assert requests_by_appraisal["unwanted"].model_dump(exclude_none=True) == {
        "utterance": "They scheduled the visit for Friday.",
        "appraisal": "unwanted",
        "response_goal": PUBLIC_GOALS["unwanted"][0],
    }
    assert candidate.model_dump() == original_candidate

    output_path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["response_goal_policy"] == script.RESPONSE_GOAL_POLICY
    assert "response_candidate_status" not in output
    pair = output["pairs"][0]
    assert pair["utterance"] == "They scheduled the visit for Friday."
    assert set(pair["conditions"]) == {"wanted", "unwanted"}
    wanted = pair["conditions"]["wanted"]
    assert wanted["private_context"]
    assert wanted["vocal_direction"]
    assert wanted["response_goals"] == PUBLIC_GOALS["wanted"]
    assert wanted["selected_goal_index"] == 0
    assert wanted["selected_response_goal"] == wanted["response_goals"][0]
    assert wanted["selected_response_goal"] == requests_by_appraisal["wanted"].response_goal
    assert wanted["response"] == {
        "text": "Response for wanted.",
        "vocal_direction": "Respond in a wanted manner.",
    }
    unwanted = pair["conditions"]["unwanted"]
    assert unwanted["response_goals"] == PUBLIC_GOALS["unwanted"]
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
    assert len(goal_calls) == 2
    assert len(response_calls) == 2


@pytest.mark.parametrize(
    ("wanted_goal_count", "unwanted_goal_count", "reverse_conditions"),
    [(3, 3, False), (2, 3, False), (3, 3, True)],
)
def test_all_mode_generates_and_stores_each_goal_by_appraisal(
    tmp_path,
    monkeypatch,
    wanted_goal_count,
    unwanted_goal_count,
    reverse_conditions,
):
    goal_calls = []
    response_calls = []
    candidate = make_pair(
        reverse_conditions=reverse_conditions,
        wanted_goal_count=wanted_goal_count,
        unwanted_goal_count=unwanted_goal_count,
    )
    original_candidate = candidate.model_dump()

    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[candidate]),
    )
    install_fake_goal_generation(monkeypatch, goal_calls)

    def fake_generate_response(_client, request, _model, _max_output_tokens):
        response_calls.append(request)
        return ResponseGenerationResult(
            response_text=f"{request.appraisal}: {request.response_goal}",
            response_vocal_direction=f"Direction for {request.appraisal}.",
        )

    monkeypatch.setattr(script, "generate_response", fake_generate_response)

    dataset = script.generate_axis(
        SimpleNamespace(),
        contrast_id="wanted_vs_unwanted",
        target_count=1,
        batch_size=1,
        max_batches=1,
        output_dir=tmp_path,
        model="test-model",
        max_output_tokens=512,
        goal_mode="all",
    )

    assert dataset.response_goal_policy == script.ALL_RESPONSE_GOALS_POLICY
    assert dataset.response_candidate_status == "unreviewed"
    assert dataset.candidate_review_calls == 1
    assert dataset.response_goal_calls == 2
    assert len(goal_calls) == 2
    assert len(response_calls) == 4
    assert candidate.model_dump() == original_candidate

    output = json.loads(
        (tmp_path / "wanted_vs_unwanted" / "dataset.json").read_text(encoding="utf-8")
    )
    assert output["response_goal_policy"] == script.ALL_RESPONSE_GOALS_POLICY
    assert output["response_candidate_status"] == "unreviewed"
    pair = output["pairs"][0]
    requests = {(request.appraisal, request.response_goal): request for request in response_calls}
    for appraisal, condition in pair["conditions"].items():
        assert "selected_goal_index" not in condition
        assert "selected_response_goal" not in condition
        assert "response" not in condition
        assert len(condition["response_candidates"]) == len(condition["response_goals"])
        for goal_index, candidate_response in enumerate(condition["response_candidates"]):
            response_goal = condition["response_goals"][goal_index]
            assert candidate_response["goal_index"] == goal_index
            assert candidate_response["response_goal"] == response_goal
            request = requests[(appraisal, response_goal)]
            assert request.model_dump(exclude_none=True) == {
                "utterance": pair["utterance"],
                "appraisal": appraisal,
                "response_goal": response_goal,
            }
            assert candidate_response["response"]["text"] == (f"{appraisal}: {response_goal}")

    resumed = script.generate_axis(
        SimpleNamespace(),
        contrast_id="wanted_vs_unwanted",
        target_count=1,
        batch_size=1,
        max_batches=1,
        output_dir=tmp_path,
        model="test-model",
        max_output_tokens=512,
        goal_mode="all",
    )
    assert resumed.completed_pair_count == 1
    assert len(goal_calls) == 2
    assert len(response_calls) == 4


@pytest.mark.parametrize(
    ("stored_mode", "requested_mode", "remove_policy"),
    [
        ("first", "first", True),
        ("first", "all", False),
        ("all", "first", False),
    ],
)
def test_resume_rejects_missing_or_different_goal_policy(
    tmp_path,
    stored_mode,
    requested_mode,
    remove_policy,
):
    path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    path.parent.mkdir(parents=True)
    stored = script._new_dataset(
        "wanted_vs_unwanted",
        1,
        "test-model",
        stored_mode,
    ).model_dump(exclude_none=True)
    if remove_policy:
        stored.pop("response_goal_policy")
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
            goal_mode=requested_mode,
        )


def test_inconsistent_event_candidate_is_skipped_before_goal_generation(
    tmp_path,
    monkeypatch,
):
    rejected = make_pair(utterance="They changed the hidden event.")
    accepted = make_pair(utterance="They scheduled the visit for Friday.")
    candidates = iter([rejected, accepted])
    reviewed_utterances = []
    goal_calls = []
    response_calls = []

    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[next(candidates)]),
    )

    def fake_event_review(_client, *, pair, **_kwargs):
        reviewed_utterances.append(pair.utterance)
        return pair is accepted

    monkeypatch.setattr(script, "_has_consistent_shared_event", fake_event_review)
    install_fake_goal_generation(monkeypatch, goal_calls)

    def fake_generate_response(_client, request, _model, _max_output_tokens):
        response_calls.append(request)
        return ResponseGenerationResult(
            response_text=f"Response for {request.appraisal}.",
            response_vocal_direction="Calm.",
        )

    monkeypatch.setattr(script, "generate_response", fake_generate_response)

    dataset = script.generate_axis(
        SimpleNamespace(),
        contrast_id="wanted_vs_unwanted",
        target_count=1,
        batch_size=1,
        max_batches=2,
        output_dir=tmp_path,
        model="test-model",
        max_output_tokens=512,
    )

    assert reviewed_utterances == [rejected.utterance, accepted.utterance]
    assert dataset.candidate_calls == 2
    assert dataset.candidate_review_calls == 2
    assert dataset.response_goal_calls == 2
    assert len(goal_calls) == 2
    assert all(request.utterance == accepted.utterance for request in goal_calls)
    assert len(response_calls) == 2
    assert dataset.pairs[0].utterance == accepted.utterance


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
    install_fake_goal_generation(monkeypatch)

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
    assert stored["candidate_review_calls"] == 0
    assert stored["response_goal_calls"] == 0
    assert stored["response_calls"] == 0
    assert stored["pairs"] == []


def test_all_mode_failure_preserves_completed_pair_only(tmp_path, monkeypatch):
    first_candidate = make_pair(
        wanted_goal_count=2,
        unwanted_goal_count=3,
        utterance="They scheduled the first visit for Friday.",
    )
    second_candidate = make_pair(
        wanted_goal_count=2,
        unwanted_goal_count=3,
        utterance="They scheduled the second visit for Friday.",
    )
    originals = [first_candidate.model_dump(), second_candidate.model_dump()]
    response_calls = []

    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[first_candidate, second_candidate]),
    )
    install_fake_goal_generation(monkeypatch)

    def fail_during_second_pair(_client, request, _model, _max_output_tokens):
        response_calls.append(request)
        if len(response_calls) == 7:
            raise RuntimeError("provider failed")
        return ResponseGenerationResult(
            response_text=f"Response {len(response_calls)}.",
            response_vocal_direction="Calm.",
        )

    monkeypatch.setattr(script, "generate_response", fail_during_second_pair)

    with pytest.raises(RuntimeError, match="provider failed"):
        script.generate_axis(
            SimpleNamespace(),
            contrast_id="wanted_vs_unwanted",
            target_count=2,
            batch_size=2,
            max_batches=1,
            output_dir=tmp_path,
            model="test-model",
            max_output_tokens=512,
            goal_mode="all",
        )

    assert len(response_calls) == 7
    assert first_candidate.model_dump() == originals[0]
    assert second_candidate.model_dump() == originals[1]
    stored = json.loads(
        (tmp_path / "wanted_vs_unwanted" / "dataset.json").read_text(encoding="utf-8")
    )
    assert stored["completed_pair_count"] == 1
    assert stored["candidate_calls"] == 1
    assert stored["candidate_review_calls"] == 1
    assert stored["response_goal_calls"] == 2
    assert stored["response_calls"] == 4
    assert len(stored["pairs"]) == 1
    assert stored["pairs"][0]["utterance"] == first_candidate.utterance
