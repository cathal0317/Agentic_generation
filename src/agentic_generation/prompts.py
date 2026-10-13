"""System prompts for the user and assistant dialogue roles."""

import json

from agentic_generation.spec import ConversationSpec

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

ASSISTANT_SYSTEM_PROMPT = """You are a helpful, respectful conversational assistant.
Speak naturally aloud.
Usually use one to three short sentences and address one point at a time.
Ask at most one focused question when useful; not every response needs a question.
Avoid headings, bullet lists, placeholders, and copy/paste instructions.
Base claims about the user only on information disclosed in the dialogue.
Match the response to that information.
Do not automatically agree, reassure, or repeat empathy phrases.
Offer practical help when relevant.
When the user is finished, acknowledge it briefly.
Do not introduce another task or repeatedly offer further help.
Return only the next spoken assistant utterance."""


def build_user_system_prompt(spec: ConversationSpec) -> str:
    """Build the user role prompt from validated scenario input."""
    personality = json.dumps(spec.user_personality, indent=2, sort_keys=True, ensure_ascii=False)
    return (
        f"Scenario:\n{spec.scenario}\n\n"
        f"User personality:\n{personality}\n\n"
        f"Instructions:\n{USER_PROMPT_INSTRUCTIONS}"
    )


def build_assistant_system_prompt() -> str:
    """Return instructions that are independent of scenario input."""
    return ASSISTANT_SYSTEM_PROMPT
