import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    "wanted": ["Acknowledge the welcome change.", "Show interest in the outcome."],
    "unwanted": ["Acknowledge the unwelcome change.", "Invite the speaker to elaborate."],
}


def make_pair(utterance="They scheduled the visit for Friday."):
    return ContrastivePair(
        utterance=utterance,
        appraisal_axis="wanted_vs_unwanted",
        conditions=[
            ContrastCondition(
                appraisal="wanted",
                private_context="Friday is the speaker's preferred day.",
                vocal_direction="Sound pleased and relaxed.",
            ),
            ContrastCondition(
                appraisal="unwanted",
                private_context="Friday conflicts with the speaker's plans.",
                vocal_direction="Sound restrained and displeased.",
            ),
        ],
    )


def install_passing_pipeline(monkeypatch, events=None, selected=(1, 0)):
    events = events if events is not None else []

    def review_pair(_client, pair, _model, _tokens):
        events.append(("pair_review", pair.utterance))
        return SimpleNamespace()

    def generate_goals(_client, request, _model, _tokens):
        events.append(("goal_generation", request.appraisal))
        return ResponseGoalsResult(response_goals=PUBLIC_GOALS[request.appraisal])

    def review_goals(_client, pair, _model, _tokens):
        events.append(("goal_review", [condition.response_goals for condition in pair.conditions]))
        return SimpleNamespace()

    def generate_response(_client, request, _model, _tokens):
        events.append(("response_generation", request.appraisal, request.response_goal))
        return ResponseGenerationResult(
            response_text=f"{request.appraisal}: {request.response_goal}",
            response_vocal_direction="Calm and attentive.",
        )

    def review_responses(_client, source, _model, _tokens):
        events.append(("response_review", len(source.conditions)))
        return SimpleNamespace()

    monkeypatch.setattr(script, "review_contrastive_pair", review_pair)
    monkeypatch.setattr(
        script,
        "determine_review_decision",
        lambda *_args: "ready_for_response_goals",
    )
    monkeypatch.setattr(script, "generate_response_goals", generate_goals)
    monkeypatch.setattr(script, "review_response_goals", review_goals)
    monkeypatch.setattr(
        script,
        "determine_response_goals_decision",
        lambda *_args: "ready_for_response_generation",
    )
    monkeypatch.setattr(script, "generate_response", generate_response)
    monkeypatch.setattr(script, "review_realized_responses", review_responses)
    monkeypatch.setattr(
        script,
        "determine_response_selection",
        lambda *_args: ("ready_for_audio_review", list(selected)),
    )
    return events


def test_dry_run_reports_each_separate_stage(tmp_path):
    output = script.build_dry_run(
        ["wanted_vs_unwanted"],
        target_count=3,
        batch_size=2,
        output_dir=tmp_path,
    )

    assert output["response_selection_policy"] == script.RESPONSE_SELECTION_POLICY
    assert output["minimum_candidate_calls"] == 2
    assert output["minimum_candidate_review_calls"] == 3
    assert output["response_goal_calls"] == 6
    assert output["response_goal_review_calls"] == 3
    assert output["response_calls"] == 12
    assert output["response_review_calls"] == 3


def test_pipeline_reviews_exact_artifacts_in_stage_order(tmp_path, monkeypatch):
    pair = make_pair()
    events = []
    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[pair]),
    )
    install_passing_pipeline(monkeypatch, events, selected=(1, 0))

    dataset = script.generate_axis(
        SimpleNamespace(),
        contrast_id="wanted_vs_unwanted",
        target_count=1,
        batch_size=1,
        max_batches=1,
        output_dir=tmp_path,
        model="test-model",
        max_output_tokens=512,
    )

    assert [event[0] for event in events] == [
        "pair_review",
        "goal_generation",
        "goal_generation",
        "goal_review",
        "response_generation",
        "response_generation",
        "response_generation",
        "response_generation",
        "response_review",
    ]
    assert dataset.candidate_review_calls == 1
    assert dataset.response_goal_calls == 2
    assert dataset.response_goal_review_calls == 1
    assert dataset.response_calls == 4
    assert dataset.response_review_calls == 1

    stored = json.loads(
        (tmp_path / "wanted_vs_unwanted" / "dataset.json").read_text(encoding="utf-8")
    )
    assert stored["pairs"][0]["status"] == "ready_for_audio_review"
    wanted = stored["pairs"][0]["conditions"]["wanted"]
    unwanted = stored["pairs"][0]["conditions"]["unwanted"]
    assert wanted["selected_goal_index"] == 1
    assert wanted["selected_response_goal"] == PUBLIC_GOALS["wanted"][1]
    assert unwanted["selected_goal_index"] == 0
    assert unwanted["selected_response_goal"] == PUBLIC_GOALS["unwanted"][0]


def test_rejected_pair_never_generates_goals(tmp_path, monkeypatch):
    rejected = make_pair("They changed the hidden event.")
    accepted = make_pair()
    candidates = iter([rejected, accepted])
    goal_calls = []

    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[next(candidates)]),
    )
    install_passing_pipeline(monkeypatch)
    monkeypatch.setattr(
        script,
        "determine_review_decision",
        lambda pair, _review: (
            "revise_candidate"
            if pair.utterance == rejected.utterance
            else "ready_for_response_goals"
        ),
    )
    original_goal_generation = script.generate_response_goals

    def record_goals(*args, **kwargs):
        goal_calls.append(args[1].utterance)
        return original_goal_generation(*args, **kwargs)

    monkeypatch.setattr(script, "generate_response_goals", record_goals)

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

    assert dataset.candidate_review_calls == 2
    assert goal_calls == [accepted.utterance, accepted.utterance]


def test_rejected_final_goals_never_generate_responses(tmp_path, monkeypatch):
    rejected = make_pair("They scheduled the first visit.")
    accepted = make_pair("They scheduled the second visit.")
    candidates = iter([rejected, accepted])
    response_utterances = []

    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[next(candidates)]),
    )
    install_passing_pipeline(monkeypatch)
    monkeypatch.setattr(
        script,
        "determine_response_goals_decision",
        lambda pair, _review: (
            "revise_response_goals"
            if pair.utterance == rejected.utterance
            else "ready_for_response_generation"
        ),
    )
    original_response_generation = script.generate_response

    def record_response(client, request, model, tokens):
        response_utterances.append(request.utterance)
        return original_response_generation(client, request, model, tokens)

    monkeypatch.setattr(script, "generate_response", record_response)

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

    assert dataset.response_goal_review_calls == 2
    assert response_utterances == [accepted.utterance] * 4


def test_failure_does_not_checkpoint_partial_pair(tmp_path, monkeypatch):
    pair = make_pair()
    path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    script._write_dataset(path, script._new_dataset("wanted_vs_unwanted", 1, "test-model"))
    monkeypatch.setattr(
        script,
        "generate_contrastive_pairs",
        lambda *_args, **_kwargs: ContrastiveBatch(pairs=[pair]),
    )
    install_passing_pipeline(monkeypatch)

    calls = 0
    original = script.generate_response

    def fail_second(client, request, model, tokens):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("provider failed")
        return original(client, request, model, tokens)

    monkeypatch.setattr(script, "generate_response", fail_second)
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

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["completed_pair_count"] == 0
    assert stored["pairs"] == []


def test_resume_rejects_old_pipeline_schema(tmp_path):
    path = tmp_path / "wanted_vs_unwanted" / "dataset.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"response_goal_policy": "first_candidate_goal_v1"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="new output directory"):
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
