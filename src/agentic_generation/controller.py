"""Shared dialogue progress assessment types and guidance."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["continue", "end", "stalled"]
TerminationReason = Literal["natural_end", "stalled", "max_exchanges"]

REVIEWER_SYSTEM_PROMPT = """Review only the visible dialogue and assess its progress.
Choose continue when the conversation develops through relevant information, clarification,
a useful response, or contextually meaningful emotional support.
Choose end when the user explicitly wants to finish or the exchange reaches a natural closing
without an outstanding request.
Choose stalled when the latest exchange repeats earlier content without meaningful development.
A thank-you followed by another question is not an ending.
Emotional acknowledgement can be meaningful without introducing new facts.
Do not require every concern to be resolved or infer undisclosed private context.
Give a short reason grounded in the visible dialogue."""


class DialogueAssessment(BaseModel):
    """A validated decision about the latest completed exchange."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    decision: Decision
    reason: str = Field(min_length=1, max_length=300)


class PublicTurn(TypedDict):
    turn_index: int
    role: Literal["user", "assistant"]
    text: str


DialogueReviewer = Callable[[list[PublicTurn]], DialogueAssessment]


@dataclass(frozen=True)
class ControlRecord:
    after_turn_index: int
    decision: Decision
    reason: str
