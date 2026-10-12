"""Validated input for a conversation generation run."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConversationSpec(BaseModel):
    """Scenario inputs that may vary without changing orchestration code."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    scenario: str = Field(min_length=1)
    user_personality: dict[str, Any] = Field(default_factory=dict)
    max_exchanges: int = Field(default=8, gt=0)
