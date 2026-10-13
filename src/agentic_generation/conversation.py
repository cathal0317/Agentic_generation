"""In-memory orchestration for alternating user and assistant turns."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

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


@dataclass(frozen=True)
class Turn:
    turn_index: int
    role: Role
    text: str


def _user_messages(turns: list[Turn]) -> list[ChatMessage]:
    messages = [ChatMessage(role="user", content=OPENING_USER_REQUEST)]
    for turn in turns:
        role: Role = "assistant" if turn.role == "user" else "user"
        messages.append(ChatMessage(role=role, content=turn.text))
    return messages


def _assistant_messages(turns: list[Turn]) -> list[ChatMessage]:
    return [ChatMessage(role=turn.role, content=turn.text) for turn in turns]


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
) -> list[Turn]:
    """Generate a fixed number of alternating exchanges in memory."""
    turns: list[Turn] = []
    user_system_prompt = build_user_system_prompt(spec)
    assistant_system_prompt = build_assistant_system_prompt()

    for _ in range(spec.max_exchanges):
        user_text = _validated_output(
            user_generator(user_system_prompt, _user_messages(turns)),
            "user",
        )
        turns.append(Turn(turn_index=len(turns), role="user", text=user_text))

        assistant_text = _validated_output(
            assistant_generator(assistant_system_prompt, _assistant_messages(turns)),
            "assistant",
        )
        turns.append(Turn(turn_index=len(turns), role="assistant", text=assistant_text))

    return turns
