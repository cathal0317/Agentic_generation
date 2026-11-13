import json
from types import SimpleNamespace

import pytest

from agentic_generation.contrastive.models import ResponseGoalCondition, ResponseGoalPair
from agentic_generation.contrastive.response_generation import ResponseGenerationResult
from agentic_generation.contrastive.response_review import (
    RESPONSE_REVIEW_SYSTEM_PROMPT,
    ConditionResponseSelection,
    RealizedResponsesReview,
    ResponseCandidateReview,
    build_realized_responses_input,
    determine_response_selection,
    review_realized_responses,
)
from agentic_generation.contrastive.review import CriterionResult, ReviewCorrespondenceError


def criterion(verdict="pass"):
    return CriterionResult(verdict=verdict, reason=f"Criterion is {verdict}.")


@pytest.fixture
def goal_pair():
    return ResponseGoalPair(
        utterance="They moved the meeting to Friday.",
        appraisal_axis="wanted_vs_unwanted",
        conditions=[
            ResponseGoalCondition(
                appraisal="wanted",
                private_context="Friday suits the speaker.",
                response_goals=["Acknowledge the welcome change.", "Show interest."],
            ),
            ResponseGoalCondition(
                appraisal="unwanted",
                private_context="Friday conflicts with the speaker's plans.",
                response_goals=["Acknowledge the unwelcome change.", "Invite elaboration."],
            ),
        ],
    )


@pytest.fixture
def responses():
    return [
        [
            ResponseGenerationResult(
                response_text=f"Response {condition_index}-{goal_index}.",
                response_vocal_direction="Calm and attentive.",
            )
            for goal_index in range(2)
        ]
        for condition_index in range(2)
    ]


def candidate(goal_index, verdict="pass"):
    return ResponseCandidateReview(
        goal_index=goal_index,
        groundedness=criterion(verdict),
        goal_alignment=criterion(),
        role_preservation=criterion(),
        natural_spoken_form=criterion(),
    )


def make_review(*, selected=(1, 0), selected_verdict="pass", consequence="pass"):
    conditions = []
    for condition_index, selected_goal_index in enumerate(selected):
        candidates = [candidate(0), candidate(1)]
        candidates[selected_goal_index] = candidate(selected_goal_index, selected_verdict)
        conditions.append(
            ConditionResponseSelection(
                condition_index=condition_index,
                selected_goal_index=selected_goal_index,
                candidates=candidates,
            )
        )
    return RealizedResponsesReview(
        conditions=conditions,
        response_consequence=criterion(consequence),
    )


def test_builds_public_review_input_without_private_context(goal_pair, responses):
    source = build_realized_responses_input(goal_pair, responses)

    payload = source.model_dump()
    assert payload["utterance"] == goal_pair.utterance
    assert "private_context" not in json.dumps(payload)
    assert payload["conditions"][0]["candidates"][1]["response_goal"] == "Show interest."
    assert payload["conditions"][0]["candidates"][1]["response_text"] == "Response 0-1."


def test_reviewer_selects_one_valid_candidate_per_condition(goal_pair, responses):
    source = build_realized_responses_input(goal_pair, responses)
    review = make_review(selected=(1, 0))

    decision, selected = determine_response_selection(source, review)

    assert decision == "ready_for_audio_review"
    assert selected == [1, 0]


def test_selection_indexes_follow_source_order_when_review_conditions_are_reordered(
    goal_pair,
    responses,
):
    source = build_realized_responses_input(goal_pair, responses)
    review = make_review(selected=(1, 0))
    review = review.model_copy(update={"conditions": list(reversed(review.conditions))})

    decision, selected = determine_response_selection(source, review)

    assert decision == "ready_for_audio_review"
    assert selected == [1, 0]


@pytest.mark.parametrize(
    ("selected_verdict", "consequence", "expected"),
    [
        ("fail", "pass", "revise_responses"),
        ("uncertain", "pass", "needs_human_review"),
        ("pass", "fail", "revise_responses"),
        ("pass", "uncertain", "needs_human_review"),
    ],
)
def test_selection_rejects_bad_selected_responses(
    goal_pair,
    responses,
    selected_verdict,
    consequence,
    expected,
):
    source = build_realized_responses_input(goal_pair, responses)
    decision, selected = determine_response_selection(
        source,
        make_review(selected_verdict=selected_verdict, consequence=consequence),
    )
    assert decision == expected
    assert selected is None


def test_review_call_uses_structured_source(goal_pair, responses):
    source = build_realized_responses_input(goal_pair, responses)
    expected = make_review()

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(status="completed", output_parsed=expected)

    client = SimpleNamespace(responses=FakeResponses())
    assert review_realized_responses(client, source, "model", 512) == expected
    call = client.responses.calls[0]
    assert call["instructions"] == RESPONSE_REVIEW_SYSTEM_PROMPT
    assert call["text_format"] is RealizedResponsesReview
    assert json.loads(call["input"][0]["content"]) == source.model_dump()


def test_rejects_missing_candidate_review(goal_pair, responses):
    source = build_realized_responses_input(goal_pair, responses)
    review = make_review()
    first = review.conditions[0].model_copy(
        update={"candidates": [review.conditions[0].candidates[0]]}
    )
    invalid = review.model_copy(update={"conditions": [first, review.conditions[1]]})

    with pytest.raises(ReviewCorrespondenceError, match="goal_index"):
        determine_response_selection(source, invalid)
