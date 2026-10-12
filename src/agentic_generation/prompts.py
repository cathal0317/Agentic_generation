"""System prompts for the user and assistant dialogue roles."""

import json

from agentic_generation.spec import ConversationSpec

USER_PROMPT_INSTRUCTIONS = """Play the user and produce only the next user utterance.
Keep the supplied facts consistent.
Reveal relevant details naturally as the conversation develops.
Respond to what the assistant actually says.
Do not recite the scenario, mention the simulation, or force a happy ending.
Return only plain dialogue text without labels, JSON, or speech metadata."""

ASSISTANT_SYSTEM_PROMPT = """You are a helpful, respectful conversational assistant.
Respond to the dialogue and ask for clarification when needed.
Do not claim to know facts that the user has not disclosed.
Return only the next assistant utterance as plain dialogue text.
Do not include labels, JSON, or speech metadata."""


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
