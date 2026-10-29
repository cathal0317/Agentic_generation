"""In-memory orchestration for alternating user and assistant turns."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict

from agentic_generation.controller import (
    ControlRecord,
    DialogueAssessment,
    DialogueReviewer,
    PublicTurn,
    TerminationReason,
)
from agentic_generation.prompts import build_assistant_system_prompt, build_user_system_prompt
from agentic_generation.spec import ConversationSpec

if TYPE_CHECKING:
    from agentic_generation.premises import ScenarioContract

Role = Literal["user", "assistant"]
GenerationStage = Literal["user", "assistant", "reviewer"]


class ChatMessage(TypedDict):
    role: Role
    content: str


TextGenerator = Callable[[str, list[ChatMessage]], str]

OPENING_USER_REQUEST = (
    "Begin with a brief, natural opening about one thing on your mind. "
    "Do not summarise your background or list the scenario’s facts."
)
CONTINUATION_USER_REQUEST = (
    "Continue from the supplied opening and respond naturally to the assistant."
)


@dataclass(frozen=True)
class Turn:
    turn_index: int
    role: Role
    text: str


@dataclass(frozen=True)
class ConversationResult:
    turns: list[Turn]
    termination_reason: TerminationReason
    control_trace: list[ControlRecord]


@dataclass
class ConversationProgress:
    """Mutable progress retained if one generation stage fails."""

    turns: list[Turn] = field(default_factory=list)
    control_trace: list[ControlRecord] = field(default_factory=list)
    current_stage: GenerationStage | None = None


def _user_messages(
    turns: list[Turn],
    has_supplied_opening: bool,
) -> list[ChatMessage]:
    request = CONTINUATION_USER_REQUEST if has_supplied_opening else OPENING_USER_REQUEST
    messages = [ChatMessage(role="user", content=request)]
    for turn in turns:
        role: Role = "assistant" if turn.role == "user" else "user"
        messages.append(ChatMessage(role=role, content=turn.text))
    return messages


def _assistant_messages(turns: list[Turn]) -> list[ChatMessage]:
    return [ChatMessage(role=turn.role, content=turn.text) for turn in turns]


def _public_transcript(turns: list[Turn]) -> list[PublicTurn]:
    return [
        PublicTurn(turn_index=turn.turn_index, role=turn.role, text=turn.text) for turn in turns
    ]


def _validated_output(output: object, role: Role) -> str:
    if not isinstance(output, str):
        raise TypeError(f"{role} generator returned a non-string output")
    if not output.strip():
        raise ValueError(f"{role} generator returned a blank output")
    return output


def generate_conversation(
    spec: ConversationSpec,
    user_generator: TextGenerator,
    assistant_generator: TextGenerator,
    reviewer: DialogueReviewer,
    opening: str | None = None,
    concern: str | None = None,
    contract: ScenarioContract | None = None,
    progress: ConversationProgress | None = None,
) -> ConversationResult:
    """Generate bounded exchanges and stop on a natural end or repeated stalls."""
    progress = progress or ConversationProgress()
    turns = progress.turns
    control_trace = progress.control_trace
    if opening is not None:
        turns.append(Turn(turn_index=0, role="user", text=_validated_output(opening, "user")))

    consecutive_stalls = 0
    user_system_prompt = build_user_system_prompt(spec, concern, contract)
    assistant_system_prompt = build_assistant_system_prompt()
    has_supplied_opening = opening is not None

    for _ in range(spec.max_exchanges):
        if not turns or turns[-1].role == "assistant":
            progress.current_stage = "user"
            user_text = _validated_output(
                user_generator(
                    user_system_prompt,
                    _user_messages(turns, has_supplied_opening),
                ),
                "user",
            )
            turns.append(Turn(turn_index=len(turns), role="user", text=user_text))

        progress.current_stage = "assistant"
        assistant_text = _validated_output(
            assistant_generator(assistant_system_prompt, _assistant_messages(turns)),
            "assistant",
        )
        turns.append(Turn(turn_index=len(turns), role="assistant", text=assistant_text))

        progress.current_stage = "reviewer"
        assessment = DialogueAssessment.model_validate(reviewer(_public_transcript(turns)))
        control_trace.append(
            ControlRecord(
                after_turn_index=turns[-1].turn_index,
                decision=assessment.decision,
                reason=assessment.reason,
            )
        )
        progress.current_stage = None

        if assessment.decision == "end":
            return ConversationResult(turns, "natural_end", control_trace)
        if assessment.decision == "stalled":
            consecutive_stalls += 1
            if consecutive_stalls == 2:
                return ConversationResult(turns, "stalled", control_trace)
        else:
            consecutive_stalls = 0

    return ConversationResult(turns, "max_exchanges", control_trace)
