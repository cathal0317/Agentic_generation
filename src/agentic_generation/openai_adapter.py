"""Synchronous OpenAI adapter for the conversation generator interface."""

from openai import OpenAI

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
