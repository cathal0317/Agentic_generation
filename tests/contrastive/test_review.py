import json
from types import SimpleNamespace

import pytest

from agentic_generation.contrastive.models import (
    ContrastCondition,
    ContrastivePair,
    ResponseGoalCondition,
    ResponseGoalPair,
)
from agentic_generation.contrastive.review import (
    CONTRASTIVE_REVIEW_SYSTEM_PROMPT,
    RESPONSE_GOALS_REVIEW_SYSTEM_PROMPT,
    ConditionResponseReview,
    ContrastiveCandidateReview,
    CriterionResult,
    GoalReview,
    PairReview,
    ResponseReview,
    ReviewCorrespondenceError,
    TextReview,
    build_pair_review_input,
    build_response_review_input,
    determine_response_goals_decision,
    determine_review_decision,
    review_contrastive_pair,
    review_response_goals,
    validate_review_correspondence,
)
from agentic_generation.openai_adapter import GenerationError


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.parse_calls = []

    def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.responses = FakeResponses(response)


@pytest.fixture
def pair():
    return ContrastivePair(
        utterance="They moved the meeting to Friday.",
        appraisal_axis="wanted_vs_unwanted",
        conditions=[
            ContrastCondition(
                appraisal="wanted",
                private_context="Friday suits the speaker.",
                vocal_direction="Light and pleased.",
            ),
            ContrastCondition(
                appraisal="unwanted",
                private_context="Friday conflicts with the speaker's plans.",
                vocal_direction="Tight and displeased.",
            ),
        ],
    )


@pytest.fixture
def goal_pair(pair):
    return ResponseGoalPair(
        utterance=pair.utterance,
        appraisal_axis=pair.appraisal_axis,
        conditions=[
            ResponseGoalCondition(
                appraisal=condition.appraisal,
                private_context=condition.private_context,
                response_goals=[
                    f"Acknowledge the {condition.appraisal} change.",
                    "Invite the speaker to elaborate.",
                ],
            )
            for condition in pair.conditions
        ],
    )


def criterion(verdict="pass"):
    return CriterionResult(verdict=verdict, reason=f"Criterion is {verdict}.")


def candidate_review(verdict="pass"):
    return ContrastiveCandidateReview(
        text=TextReview(
            event_clarity=criterion(verdict),
            appraisal_openness=criterion(),
            no_lexical_leakage=criterion(),
            natural_spoken_form=criterion(),
        ),
        pair=PairReview(
            event_consistency=criterion(),
            natural_contrast=criterion(),
            vocal_contrast_plausibility=criterion(),
            no_hidden_fact_dependency=criterion(),
        ),
    )


def goals_review(verdict="pass"):
    return ResponseReview(
        conditions=[
            ConditionResponseReview(
                condition_index=condition_index,
                goals=[
                    GoalReview(
                        goal_index=goal_index,
                        groundedness=criterion(verdict),
                        single_action=criterion(),
                        assistant_role_scope=criterion(),
                    )
                    for goal_index in range(2)
                ],
                non_prescriptive_response_space=criterion(),
            )
            for condition_index in range(2)
        ],
        response_consequence=criterion(),
    )


def test_candidate_review_payload_excludes_response_goals(pair):
    expected = candidate_review()
    client = FakeClient(SimpleNamespace(status="completed", output_parsed=expected))

    result = review_contrastive_pair(client, pair, "review-model", 768)

    assert result == expected
    call = client.responses.parse_calls[0]
    assert call["instructions"] == CONTRASTIVE_REVIEW_SYSTEM_PROMPT
    assert call["text_format"] is ContrastiveCandidateReview
    payload = json.loads(call["input"][0]["content"])
    assert set(payload) == {"text", "pair"}
    assert "response_goals" not in json.dumps(payload)
    assert payload["pair"] == build_pair_review_input(pair).model_dump()


def test_final_goal_review_uses_separately_generated_goals(goal_pair):
    expected = goals_review()
    client = FakeClient(SimpleNamespace(status="completed", output_parsed=expected))

    result = review_response_goals(client, goal_pair, "review-model", 768)

    assert result == expected
    call = client.responses.parse_calls[0]
    assert call["instructions"] == RESPONSE_GOALS_REVIEW_SYSTEM_PROMPT
    assert call["text_format"] is ResponseReview
    payload = json.loads(call["input"][0]["content"])
    assert payload == build_response_review_input(goal_pair).model_dump()
    assert "response_goals" in json.dumps(payload)
    assert "vocal_direction" not in json.dumps(payload)


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("pass", "ready_for_response_goals"),
        ("fail", "revise_candidate"),
        ("uncertain", "needs_human_review"),
    ],
)
def test_candidate_decision_is_independent_of_goals(pair, verdict, expected):
    assert determine_review_decision(pair, candidate_review(verdict)) == expected


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("pass", "ready_for_response_generation"),
        ("fail", "revise_response_goals"),
        ("uncertain", "needs_human_review"),
    ],
)
def test_goal_decision_applies_to_final_goals(goal_pair, verdict, expected):
    assert determine_response_goals_decision(goal_pair, goals_review(verdict)) == expected


def test_goal_review_requires_exact_condition_and_goal_indexes(goal_pair):
    review = goals_review()
    invalid_condition = review.conditions[0].model_copy(update={"condition_index": 2})
    invalid = review.model_copy(update={"conditions": [invalid_condition, review.conditions[1]]})
    with pytest.raises(ReviewCorrespondenceError, match="condition_index"):
        validate_review_correspondence(goal_pair, invalid)

    first = review.conditions[0]
    invalid_goal = first.goals[0].model_copy(update={"goal_index": 2})
    invalid_first = first.model_copy(update={"goals": [invalid_goal, first.goals[1]]})
    invalid = review.model_copy(update={"conditions": [invalid_first, review.conditions[1]]})
    with pytest.raises(ReviewCorrespondenceError, match="goal_index"):
        validate_review_correspondence(goal_pair, invalid)


@pytest.mark.parametrize("review_kind", ["candidate", "goals"])
def test_review_rejects_non_completed_provider_output(pair, goal_pair, review_kind):
    parsed = candidate_review() if review_kind == "candidate" else goals_review()
    client = FakeClient(SimpleNamespace(status="failed", output_parsed=parsed))

    with pytest.raises(GenerationError, match="No complete"):
        if review_kind == "candidate":
            review_contrastive_pair(client, pair, "model", 256)
        else:
            review_response_goals(client, goal_pair, "model", 256)
