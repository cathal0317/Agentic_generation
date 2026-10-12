"""Persistent conversation and turn models."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentic_generation.db import Base


class Conversation(Base):
    """A generated conversation and the inputs used to create it."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    scenario: Mapped[str] = mapped_column(Text)
    user_personality: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Private generator inputs belong here, not in initial_context.
    initial_condition: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Only information explicitly given to the assistant belongs here.
    initial_context: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    turns: Mapped[list["ConversationTurn"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ConversationTurn.turn_index",
    )


class ConversationTurn(Base):
    """One ordered user or assistant message in a conversation."""

    __tablename__ = "conversation_turns"
    __table_args__ = (
        UniqueConstraint("conversation_id", "turn_index"),
        CheckConstraint("turn_index >= 0", name="turn_index_non_negative"),
        CheckConstraint("length(btrim(text)) > 0", name="text_non_blank"),
        CheckConstraint("role IN ('user', 'assistant')", name="role_allowed"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    turn_index: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")
