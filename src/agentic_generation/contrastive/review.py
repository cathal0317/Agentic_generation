"""Contracts and deterministic decisions for contrastive candidate review.

The pair model stores one shared utterance, so text invariance is structural
rather than another semantic criterion. Actual synthesized speech still
requires later audio review. Vocal contrast here is only a textual plausibility
assessment and does not establish tone sufficiency.

An appraisal is the speaker's current stance. It does not imply that the
speaker previously requested the event or identify which aspect caused the
stance. A negative stance toward a package arriving does not establish dislike
of the item itself. Asking about an unknown fact is allowed, while a question
such as "Which schedule does it conflict with?" presupposes a conflict and may
fail groundedness.

Private context is allowed as reviewer evidence and is not itself a defect.
Questions and empathy are valid moves, but neither is mandatory for every
condition. A goal that combines empathy, inquiry, and problem-solving contains
multiple conversational actions.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_generation.contrastive.models import ContrastivePair

NonEmptyText = Annotated[str, Field(min_length=1)]
Verdict = Literal["pass", "fail", "uncertain"]
ReviewDecision = Literal[
    "revise_candidate",
    "revise_response_goals",
    "needs_human_review",
    "ready_for_audio_review",
]


class TextReviewInput(BaseModel):
    """Public text shown to the text reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    utterance: NonEmptyText


class PairReviewConditionInput(BaseModel):
    """Private condition data allowed for pair-level comparison."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    appraisal: NonEmptyText
    private_context: NonEmptyText
    vocal_direction: NonEmptyText


class PairReviewInput(BaseModel):
    """Shared text and private delivery conditions, excluding response goals."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    utterance: NonEmptyText
    conditions: list[PairReviewConditionInput] = Field(min_length=2, max_length=2)


class ResponseReviewConditionInput(BaseModel):
    """Response information with no private context or vocal direction."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    appraisal: NonEmptyText
    response_goals: list[NonEmptyText] = Field(min_length=1)


class ResponseReviewInput(BaseModel):
    """Public text and appraisal-conditioned response directions."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    utterance: NonEmptyText
    conditions: list[ResponseReviewConditionInput] = Field(min_length=2, max_length=2)


def build_text_review_input(pair: ContrastivePair) -> TextReviewInput:
    """Project only the shared utterance for text review."""
    return TextReviewInput(utterance=pair.utterance)


def build_pair_review_input(pair: ContrastivePair) -> PairReviewInput:
    """Explicitly copy only fields allowed in pair review."""
    return PairReviewInput(
        utterance=pair.utterance,
        conditions=[
            PairReviewConditionInput(
                appraisal=condition.appraisal,
                private_context=condition.private_context,
                vocal_direction=condition.vocal_direction,
            )
            for condition in pair.conditions
        ],
    )


def build_response_review_input(pair: ContrastivePair) -> ResponseReviewInput:
    """Explicitly copy only public response-review fields."""
    return ResponseReviewInput(
        utterance=pair.utterance,
        conditions=[
            ResponseReviewConditionInput(
                appraisal=condition.appraisal,
                response_goals=list(condition.response_goals),
            )
            for condition in pair.conditions
        ],
    )


class CriterionResult(BaseModel):
    """One required semantic judgment and its reviewer rationale."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    verdict: Verdict
    reason: NonEmptyText
    evidence: NonEmptyText | None = None


class TextReview(BaseModel):
    """Semantic review of the shared utterance alone."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_clarity: CriterionResult = Field(
        description="Whether the utterance itself makes the event understandable."
    )
    appraisal_openness: CriterionResult = Field(
        description="Whether the text leaves the speaker's positive or negative stance open."
    )
    no_lexical_leakage: CriterionResult = Field(
        description="Whether no wording directly fixes an emotion or appraisal."
    )
    natural_spoken_form: CriterionResult = Field(
        description="Whether the utterance is plausible natural conversational speech."
    )


class PairReview(BaseModel):
    """Semantic comparison of the two private delivery conditions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_consistency: CriterionResult = Field(
        description=(
            "Whether both private contexts preserve the same objective event and outcome; "
            "preferences may differ, but outcomes may not."
        )
    )
    natural_contrast: CriterionResult = Field(
        description="Whether both appraisals apply naturally to the same utterance."
    )
    vocal_contrast_plausibility: CriterionResult = Field(
        description=(
            "Whether the written delivery directions plausibly express different stances; "
            "this is not actual audio validation."
        )
    )
    no_hidden_fact_dependency: CriterionResult = Field(
        description=(
            "Whether distinguishing the stances does not require reconstructing inaudible "
            "specific facts. Merely having private context is not a failure."
        )
    )


class GoalReview(BaseModel):
    """Review of one response goal selected by explicit source index."""

    model_config = ConfigDict(extra="forbid", strict=True)

    goal_index: int = Field(ge=0)
    groundedness: CriterionResult = Field(
        description=(
            "Whether the goal follows from the utterance and given current appraisal without "
            "assuming its unknown cause. Open questions are allowed; presuppositions are not."
        )
    )
    single_action: CriterionResult = Field(
        description="Whether the goal contains one independently selectable conversational move."
    )
    assistant_role_scope: CriterionResult = Field(
        description="Whether the goal assumes no physical action, external tool, or shared role."
    )


class ConditionResponseReview(BaseModel):
    """Review of all goals belonging to one indexed condition."""

    model_config = ConfigDict(extra="forbid", strict=True)

    condition_index: int = Field(ge=0)
    goals: list[GoalReview] = Field(min_length=1)
    non_prescriptive_response_space: CriterionResult = Field(
        description=(
            "Whether the goals permit multiple natural responses instead of prescribing exact "
            "wording or requiring the same empathy-plus-question pattern."
        )
    )


class ResponseReview(BaseModel):
    """Review of response directions without access to private condition facts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    conditions: list[ConditionResponseReview] = Field(min_length=1)
    response_consequence: CriterionResult = Field(
        description=(
            "Whether the two appraisals lead to meaningfully different appropriate response "
            "directions; some overlap is allowed."
        )
    )


class ContrastiveReview(BaseModel):
    """Complete semantic review; final disposition is computed separately."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: TextReview
    pair: PairReview
    response: ResponseReview


class ReviewCorrespondenceError(ValueError):
    """Review indexes do not correspond exactly to the source pair."""


def validate_review_correspondence(
    pair: ContrastivePair,
    review: ContrastiveReview,
) -> None:
    """Require every source condition and goal to be reviewed exactly once."""
    expected_conditions = list(range(len(pair.conditions)))
    actual_conditions = [
        condition_review.condition_index for condition_review in review.response.conditions
    ]
    if sorted(actual_conditions) != expected_conditions:
        raise ReviewCorrespondenceError(
            "condition_index values must cover each source condition exactly once"
        )

    reviews_by_condition = {
        condition_review.condition_index: condition_review
        for condition_review in review.response.conditions
    }
    for condition_index, condition in enumerate(pair.conditions):
        expected_goals = list(range(len(condition.response_goals)))
        actual_goals = [
            goal_review.goal_index for goal_review in reviews_by_condition[condition_index].goals
        ]
        if sorted(actual_goals) != expected_goals:
            raise ReviewCorrespondenceError(
                "goal_index values must cover each source goal exactly once"
            )


def _candidate_criteria(review: ContrastiveReview) -> tuple[CriterionResult, ...]:
    return (
        review.text.event_clarity,
        review.text.appraisal_openness,
        review.text.no_lexical_leakage,
        review.text.natural_spoken_form,
        review.pair.event_consistency,
        review.pair.natural_contrast,
        review.pair.vocal_contrast_plausibility,
        review.pair.no_hidden_fact_dependency,
    )


def _response_criteria(review: ContrastiveReview) -> tuple[CriterionResult, ...]:
    criteria = [review.response.response_consequence]
    for condition in review.response.conditions:
        criteria.append(condition.non_prescriptive_response_space)
        for goal in condition.goals:
            criteria.extend(
                (
                    goal.groundedness,
                    goal.single_action,
                    goal.assistant_role_scope,
                )
            )
    return tuple(criteria)


def determine_review_decision(
    pair: ContrastivePair,
    review: ContrastiveReview,
) -> ReviewDecision:
    """Validate correspondence and compute the prioritized next action.

    ``ready_for_audio_review`` is not final dataset approval. Any revision must
    be reviewed again, and synthesized speech still needs audio and human review.
    """
    validate_review_correspondence(pair, review)
    candidate_criteria = _candidate_criteria(review)
    response_criteria = _response_criteria(review)

    if any(result.verdict == "fail" for result in candidate_criteria):
        return "revise_candidate"
    if any(result.verdict == "fail" for result in response_criteria):
        return "revise_response_goals"
    if any(result.verdict == "uncertain" for result in (*candidate_criteria, *response_criteria)):
        return "needs_human_review"
    return "ready_for_audio_review"
