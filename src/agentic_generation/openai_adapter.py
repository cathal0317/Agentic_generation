"""Synchronous OpenAI adapter for the conversation generator interface."""

import json

from openai import OpenAI

from agentic_generation.controller import (
    REVIEWER_SYSTEM_PROMPT,
    DialogueAssessment,
    DialogueReviewer,
)
from agentic_generation.conversation import TextGenerator


class GenerationError(RuntimeError):
    """The provider returned no usable completed generation."""


def create_text_generator(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
) -> TextGenerator:
    """Create a model-specific generator backed by the Responses API."""

    def generate(system_prompt, messages):
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=messages,
            store=False,
            max_output_tokens=max_output_tokens,
        )
        if response.status != "completed":
            raise GenerationError("OpenAI response did not complete")
        if not isinstance(response.output_text, str) or not response.output_text.strip():
            raise GenerationError("OpenAI response contained no text")
        return response.output_text

    return generate


def create_dialogue_reviewer(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
) -> DialogueReviewer:
    """Create a structured reviewer that sees only the public transcript."""

    def review(transcript):
        response = client.responses.parse(
            model=model,
            instructions=REVIEWER_SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": json.dumps(transcript, ensure_ascii=False),
                }
            ],
            text_format=DialogueAssessment,
            store=False,
            max_output_tokens=max_output_tokens,
        )
        if response.status != "completed":
            raise GenerationError("OpenAI review did not complete")
        if not isinstance(response.output_parsed, DialogueAssessment):
            raise GenerationError("OpenAI review contained no valid assessment")
        return response.output_parsed

    return review
