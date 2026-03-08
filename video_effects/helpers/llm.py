"""Anthropic API wrapper for structured LLM calls."""

import json
import logging
from pathlib import Path

import anthropic

from video_effects.config import settings

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def get_client() -> anthropic.Anthropic:
    """Get an Anthropic client instance."""
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = _PROMPT_DIR / name
    return path.read_text()


def call_structured(
    system_prompt: str,
    user_message: str,
    response_model: type,
    model: str | None = None,
    max_tokens: int = 4096,
) -> dict:
    """Call Claude with structured output via tool use.

    Args:
        system_prompt: System prompt text.
        user_message: User message content.
        response_model: Pydantic model class for the response schema.
        model: Model ID override. Defaults to settings.LLM_MODEL.
        max_tokens: Max tokens in response.

    Returns:
        Parsed dict matching the response_model schema.
    """
    client = get_client()
    model = model or settings.LLM_MODEL

    # Build tool definition from Pydantic schema
    schema = response_model.model_json_schema()
    tool_name = "structured_output"
    tool = {
        "name": tool_name,
        "description": f"Return structured {response_model.__name__} response",
        "input_schema": schema,
    }

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool_name},
    )

    # Extract tool use result
    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError("LLM did not return structured output via tool use")


def call_text(
    system_prompt: str,
    user_message: str,
    model: str | None = None,
    max_tokens: int = 8192,
) -> str:
    """Call Claude and return raw text response (for code generation).

    Args:
        system_prompt: System prompt text.
        user_message: User message content.
        model: Model ID override. Defaults to settings.LLM_MODEL.
        max_tokens: Max tokens in response.

    Returns:
        Raw text string from the model.
    """
    client = get_client()
    model = model or settings.LLM_MODEL

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "text":
            return block.text

    raise ValueError("LLM did not return text output")
