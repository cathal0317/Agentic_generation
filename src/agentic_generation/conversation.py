"""In-memory orchestration for alternating user and assistant turns."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

from agentic_generation.controller import (
    ControlRecord,
    DialogueAssessment,
    DialogueReviewer,
    PublicTurn,
    TerminationReason,
)
from agentic_generation.prompts import build_assistant_system_prompt, build_user_system_prompt
from agentic_generation.spec import ConversationSpec

Role = Literal["user", "assistant"]


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
) -> ConversationResult:
    """Generate bounded exchanges and stop on a natural end or repeated stalls."""
    turns = (
        [Turn(turn_index=0, role="user", text=_validated_output(opening, "user"))]
        if opening is not None
        else []
    )
    control_trace: list[ControlRecord] = []
    consecutive_stalls = 0
    user_system_prompt = build_user_system_prompt(spec, concern)
    assistant_system_prompt = build_assistant_system_prompt()
    has_supplied_opening = opening is not None

    for _ in range(spec.max_exchanges):
        if not turns or turns[-1].role == "assistant":
            user_text = _validated_output(
                user_generator(
                    user_system_prompt,
                    _user_messages(turns, has_supplied_opening),
                ),
                "user",
            )
            turns.append(Turn(turn_index=len(turns), role="user", text=user_text))

        assistant_text = _validated_output(
            assistant_generator(assistant_system_prompt, _assistant_messages(turns)),
            "assistant",
        )
        turns.append(Turn(turn_index=len(turns), role="assistant", text=assistant_text))

        assessment = DialogueAssessment.model_validate(reviewer(_public_transcript(turns)))
        control_trace.append(
            ControlRecord(
                after_turn_index=turns[-1].turn_index,
                decision=assessment.decision,
                reason=assessment.reason,
            )
        )

        if assessment.decision == "end":
            return ConversationResult(turns, "natural_end", control_trace)
        if assessment.decision == "stalled":
            consecutive_stalls += 1
            if consecutive_stalls == 2:
                return ConversationResult(turns, "stalled", control_trace)
        else:
            consecutive_stalls = 0

    return ConversationResult(turns, "max_exchanges", control_trace)
