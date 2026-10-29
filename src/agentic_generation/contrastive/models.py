"""Validated contrastive appraisal data."""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyText = Annotated[str, Field(min_length=1)]


class ContrastCondition(BaseModel):
    """One vocal appraisal and its valid response directions."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    appraisal: NonEmptyText
    private_context: NonEmptyText
    vocal_direction: NonEmptyText
    response_goals: list[NonEmptyText] = Field(min_length=1)


class ContrastivePair(BaseModel):
    """One utterance interpreted through two distinct vocal appraisals."""

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
