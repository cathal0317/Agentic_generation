import json

import pytest
from pydantic import ValidationError

from agentic_generation.contrastive.models import ContrastCondition, ContrastivePair
from agentic_generation.contrastive.review import (
    ConditionResponseReview,
    ContrastiveReview,
    CriterionResult,
    GoalReview,
    PairReview,
    ResponseReview,
    ReviewCorrespondenceError,
    TextReview,
    build_pair_review_input,
    build_response_review_input,
    build_text_review_input,
    determine_review_decision,
    validate_review_correspondence,
)

PRIVATE_MARKER = "PRIVATE_MARKER_K7"
VOCAL_MARKER = "VOCAL_MARKER_Q4"
GOAL_MARKER = "GOAL_MARKER_M9"


@pytest.fixture
def pair():
    return ContrastivePair(
        utterance="They moved the meeting to Friday.",
        appraisal_axis="wanted_vs_unwanted",
        conditions=[
            ContrastCondition(
                appraisal="wanted",
                private_context=f"{PRIVATE_MARKER}_0 The new date suits the speaker.",
                vocal_direction=f"{VOCAL_MARKER}_0 Light and pleased.",
                response_goals=[
                    f"{GOAL_MARKER}_0_0 Acknowledge that the change seems welcome.",
                    f"{GOAL_MARKER}_0_1 Show interest in their reaction to the change.",
                ],
            ),
            ContrastCondition(
                appraisal="unwanted",
                private_context=f"{PRIVATE_MARKER}_1 The old date suited the speaker.",
                vocal_direction=f"{VOCAL_MARKER}_1 Tight and displeased.",
                response_goals=[
                    f"{GOAL_MARKER}_1_0 Acknowledge that the change seems unwelcome.",
                    f"{GOAL_MARKER}_1_1 Ask what about the change concerns them.",
                ],
            ),
        ],
    )


def criterion(verdict="pass"):
    return CriterionResult(
        verdict=verdict,
        reason=f"Criterion is {verdict}.",
    )


def make_review():
    return ContrastiveReview(
        text=TextReview(
            event_clarity=criterion(),
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
        response=ResponseReview(
            conditions=[
                ConditionResponseReview(
                    condition_index=condition_index,
                    goals=[
                        GoalReview(
                            goal_index=goal_index,
                            groundedness=criterion(),
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
        ),
    )


def with_text_criterion(review, field, verdict):
    return review.model_copy(
        update={
            "text": review.text.model_copy(update={field: criterion(verdict)}),
        }
    )


def with_pair_criterion(review, field, verdict):
    return review.model_copy(
        update={
            "pair": review.pair.model_copy(update={field: criterion(verdict)}),
        }
    )


def with_response_consequence(review, verdict):
    return review.model_copy(
        update={
            "response": review.response.model_copy(
                update={"response_consequence": criterion(verdict)}
            ),
        }
    )


def with_goal_criterion(review, condition_index, goal_index, field, verdict):
    conditions = list(review.response.conditions)
    goals = list(conditions[condition_index].goals)
    goals[goal_index] = goals[goal_index].model_copy(update={field: criterion(verdict)})
    conditions[condition_index] = conditions[condition_index].model_copy(update={"goals": goals})
    return review.model_copy(
        update={
            "response": review.response.model_copy(update={"conditions": conditions}),
        }
    )


def test_projections_include_only_allowed_fields_and_preserve_order(pair):
    text_input = build_text_review_input(pair)
    pair_input = build_pair_review_input(pair)
    response_input = build_response_review_input(pair)

    assert text_input.model_dump() == {"utterance": "They moved the meeting to Friday."}
    assert pair_input.model_dump() == {
        "utterance": pair.utterance,
        "conditions": [
            {
                "appraisal": condition.appraisal,
                "private_context": condition.private_context,
                "vocal_direction": condition.vocal_direction,
            }
            for condition in pair.conditions
        ],
    }
    assert response_input.model_dump() == {
        "utterance": pair.utterance,
        "conditions": [
            {
                "appraisal": condition.appraisal,
                "response_goals": condition.response_goals,
            }
            for condition in pair.conditions
        ],
    }


def test_response_projection_excludes_private_and_vocal_markers(pair):
    serialized = json.dumps(build_response_review_input(pair).model_dump())

    assert PRIVATE_MARKER not in serialized
    assert VOCAL_MARKER not in serialized
    assert GOAL_MARKER in serialized


def test_pair_projection_excludes_response_goals(pair):
    serialized = json.dumps(build_pair_review_input(pair).model_dump())

    assert GOAL_MARKER not in serialized
    assert "response_goals" not in serialized
    assert PRIVATE_MARKER in serialized
    assert VOCAL_MARKER in serialized


@pytest.mark.parametrize(
    "model_data",
    [
        {
            "model": CriterionResult,
            "data": {"verdict": "pass", "reason": "   "},
        },
        {
            "model": CriterionResult,
            "data": {"verdict": "approved", "reason": "Not an allowed verdict."},
        },
        {
            "model": TextReview,
            "data": {
                "appraisal_openness": criterion().model_dump(),
                "no_lexical_leakage": criterion().model_dump(),
                "natural_spoken_form": criterion().model_dump(),
            },
        },
        {
            "model": PairReview,
            "data": {
                "event_consistency": criterion().model_dump(),
                "natural_contrast": criterion().model_dump(),
                "vocal_contrast_plausibility": criterion().model_dump(),
            },
        },
    ],
)
def test_review_models_reject_invalid_or_missing_required_results(model_data):
    with pytest.raises(ValidationError):
        model_data["model"].model_validate(model_data["data"])


@pytest.mark.parametrize(
    "condition_indexes",
    [
        [0],
        [0, 0],
        [0, 2],
    ],
)
def test_rejects_missing_duplicate_or_out_of_range_condition_indexes(pair, condition_indexes):
    review = make_review()
    conditions = [
        review.response.conditions[index if index < 2 else 0].model_copy(
            update={"condition_index": index}
        )
        for index in condition_indexes
    ]
    review = review.model_copy(
        update={"response": review.response.model_copy(update={"conditions": conditions})}
    )

    with pytest.raises(ReviewCorrespondenceError, match="condition_index"):
        validate_review_correspondence(pair, review)


@pytest.mark.parametrize(
    "goal_indexes",
    [
        [0],
        [0, 0],
        [0, 2],
    ],
)
def test_rejects_missing_duplicate_or_out_of_range_goal_indexes(pair, goal_indexes):
    review = make_review()
    first_condition = review.response.conditions[0]
    goals = [
        first_condition.goals[index if index < 2 else 0].model_copy(update={"goal_index": index})
        for index in goal_indexes
    ]
    conditions = [
        first_condition.model_copy(update={"goals": goals}),
        review.response.conditions[1],
    ]
    review = review.model_copy(
        update={"response": review.response.model_copy(update={"conditions": conditions})}
    )

    with pytest.raises(ReviewCorrespondenceError, match="goal_index"):
        validate_review_correspondence(pair, review)


def test_explicit_indexes_allow_reordered_review_lists(pair):
    review = make_review()
    conditions = []
    for condition in reversed(review.response.conditions):
        conditions.append(condition.model_copy(update={"goals": list(reversed(condition.goals))}))
    review = review.model_copy(
        update={"response": review.response.model_copy(update={"conditions": conditions})}
    )

    assert determine_review_decision(pair, review) == "ready_for_audio_review"


def test_all_pass_is_ready_for_audio_review(pair):
    assert determine_review_decision(pair, make_review()) == "ready_for_audio_review"


@pytest.mark.parametrize(
    "review",
    [
        with_text_criterion(make_review(), "event_clarity", "fail"),
        with_pair_criterion(make_review(), "event_consistency", "fail"),
    ],
)
def test_text_or_pair_failure_revises_candidate(pair, review):
    assert determine_review_decision(pair, review) == "revise_candidate"


@pytest.mark.parametrize(
    "review",
    [
        with_response_consequence(make_review(), "fail"),
        with_goal_criterion(make_review(), 0, 0, "groundedness", "fail"),
        with_goal_criterion(make_review(), 0, 0, "single_action", "fail"),
    ],
)
def test_response_failure_revises_response_goals(pair, review):
    assert determine_review_decision(pair, review) == "revise_response_goals"


def test_uncertain_without_failure_needs_human_review(pair):
    review = with_pair_criterion(
        make_review(),
        "vocal_contrast_plausibility",
        "uncertain",
    )

    assert determine_review_decision(pair, review) == "needs_human_review"


def test_candidate_failure_has_priority_over_response_failure_and_uncertainty(pair):
    review = with_text_criterion(make_review(), "natural_spoken_form", "fail")
    review = with_response_consequence(review, "fail")
    review = with_goal_criterion(
        review,
        0,
        0,
        "assistant_role_scope",
        "uncertain",
    )

    assert determine_review_decision(pair, review) == "revise_candidate"


def test_response_failure_has_priority_over_candidate_uncertainty(pair):
    review = with_pair_criterion(make_review(), "natural_contrast", "uncertain")
    review = with_response_consequence(review, "fail")

    assert determine_review_decision(pair, review) == "revise_response_goals"


def test_projection_validation_and_decision_do_not_modify_source_pair(pair):
    original = pair.model_dump()
    review = make_review()

    build_text_review_input(pair)
    build_pair_review_input(pair)
    build_response_review_input(pair)
    validate_review_correspondence(pair, review)
    determine_review_decision(pair, review)

    assert pair.model_dump() == original
