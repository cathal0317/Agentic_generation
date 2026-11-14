"""Validated contrastive appraisal data."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyText = Annotated[str, Field(min_length=1)]


class ContrastCondition(BaseModel):
    """One appraisal-conditioned delivery of the shared utterance."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    appraisal: NonEmptyText
    private_context: NonEmptyText
    vocal_direction: NonEmptyText


class ContrastivePair(BaseModel):
    """One utterance interpreted through two distinct vocal appraisals.

    Assistant response goals are deliberately generated and reviewed only after
    this pair has passed its text and appraisal review.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    utterance: NonEmptyText
    appraisal_axis: NonEmptyText
    conditions: list[ContrastCondition] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def appraisals_are_distinct(self) -> Self:
        normalized = {
            " ".join(condition.appraisal.casefold().split()) for condition in self.conditions
        }
        if len(normalized) != len(self.conditions):
            raise ValueError("conditions must have distinct appraisals")
        return self


class ResponseGoalCondition(BaseModel):
    """Public response directions generated after a contrastive pair is accepted."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    appraisal: NonEmptyText
    private_context: NonEmptyText
    response_goals: list[NonEmptyText] = Field(min_length=1)


class ResponseGoalPair(BaseModel):
    """A reviewed contrastive pair augmented with separately generated goals."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    utterance: NonEmptyText
    appraisal_axis: NonEmptyText
    conditions: list[ResponseGoalCondition] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def appraisals_are_distinct(self) -> Self:
        normalized = {
            " ".join(condition.appraisal.casefold().split()) for condition in self.conditions
        }
        if len(normalized) != len(self.conditions):
            raise ValueError("conditions must have distinct appraisals")
        return self
