"""Validated conversation premise proposals for human review."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

PROMPT_VERSION = "premises-v1"
NonEmptyText = Annotated[str, Field(min_length=1)]


class Premise(BaseModel):
    """One grounded concern and a possible spoken opening."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    concern: NonEmptyText
    opening: NonEmptyText
    source_quotes: list[NonEmptyText] = Field(min_length=1)


class PremiseBatch(BaseModel):
    """One to six premise proposals returned for human review."""

    model_config = ConfigDict(extra="forbid", strict=True)

    premises: list[Premise] = Field(min_length=1, max_length=6)
