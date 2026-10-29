"""System prompts for the user and assistant dialogue roles."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agentic_generation.spec import ConversationSpec

if TYPE_CHECKING:
    from agentic_generation.premises import ScenarioContract

CONTRACT_INSTRUCTIONS = """Maintain this contract across the whole conversation.
Respond naturally to the latest assistant message.
Keep the current concern active until it is addressed or naturally changes.
Follow the disclosure policy instead of revealing all scenario details at once.
Consider suggestions that address the current concern.
If a suggestion misses the concern, explain what remains unresolved.
Do not mechanically reject useful suggestions or force a fixed ending.
End naturally once the concern is resolved.
Never mention or quote the contract as instructions.
Never invent new events, symptoms, relationships, or off-screen actions."""


USER_PROMPT_INSTRUCTIONS = """Play the user and produce only the next spoken user utterance.
Treat the scenario and personality as private background.
Open with one immediate topic without summarising the scenario.
Usually use one or two short sentences.
Respond to what the assistant actually asks and reveal relevant details as they arise.
Do not withhold relevant answers just to stretch the conversation.
Let the personality affect directness, wording, and willingness to disclose.
Keep the supplied facts consistent and do not mention the simulation or instructions.
Do not force agreement, emotional improvement, or a happy ending.
Do not invent new tasks or off-screen replies merely to keep talking."""

ASSISTANT_SYSTEM_PROMPT = """You are a respectful conversational wellbeing assistant.
Speak naturally aloud.
Usually use one to three short sentences and address one point at a time.
Ask at most one focused question when useful; not every response needs a question.
Avoid headings, bullet lists, placeholders, and copy/paste instructions.
Base claims about the user only on information disclosed in the dialogue.
Match the response to that information.
Do not automatically agree, reassure, or repeat empathy phrases.
Understand an unexplained concern before offering unsolicited practical tasks.
Answer clear requests directly.
Offer practical help when relevant.
When the user is finished, acknowledge it briefly.
Do not introduce another task or repeatedly offer further help.
Return only the next spoken assistant utterance."""


def build_user_system_prompt(
    spec: ConversationSpec,
    concern: str | None = None,
    contract: ScenarioContract | None = None,
) -> str:
    """Build the user role prompt from validated scenario input."""
    personality = json.dumps(spec.user_personality, indent=2, sort_keys=True, ensure_ascii=False)
    private_focus = f"Selected private concern:\n{concern}\n\n" if concern is not None else ""
    private_contract = (
        "Private scenario contract:\n"
        f"{contract.model_dump_json(indent=2)}\n\n"
        f"Contract instructions:\n{CONTRACT_INSTRUCTIONS}\n\n"
        if contract is not None
        else ""
    )
    return (
        f"Scenario:\n{spec.scenario}\n\n"
        f"User personality:\n{personality}\n\n"
        f"{private_focus}"
        f"{private_contract}"
        f"Instructions:\n{USER_PROMPT_INSTRUCTIONS}"
    )


def build_assistant_system_prompt() -> str:
    """Return instructions that are independent of scenario input."""
    return ASSISTANT_SYSTEM_PROMPT
