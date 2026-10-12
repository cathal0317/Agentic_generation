"""Unit test for conversation storage on disposable PostgreSQL."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agentic_generation.models import Conversation, ConversationTurn

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_conversation_storage_constraints_and_migration_cycle():
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL + pgvector database")

    engine = create_engine(database_url, connect_args={"connect_timeout": 5})
    config = Config(str(ROOT / "alembic.ini"))

    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        assert {"conversations", "conversation_turns"} <= set(inspect(engine).get_table_names())

        conversation = Conversation(
            scenario="A traveler asks for directions.",
            user_personality={"tone": "curious"},
            initial_condition={"destination": "museum", "generator_seed": 17},
            initial_context="The assistant was told that the user is near the station.",
            turns=[
                ConversationTurn(turn_index=2, role="user", text="Is it far from here?"),
                ConversationTurn(turn_index=0, role="user", text="How do I reach the museum?"),
                ConversationTurn(
                    turn_index=3,
                    role="assistant",
                    text="No, it is about a ten-minute walk.",
                ),
                ConversationTurn(
                    turn_index=1,
                    role="assistant",
                    text="Walk north for two blocks, then turn right.",
                ),
            ],
        )

        with Session(engine) as session:
            session.add(conversation)
            session.commit()
            conversation_id = conversation.id

        with Session(engine) as session:
            stored = session.get(Conversation, conversation_id)
            assert stored is not None
            assert stored.initial_condition["generator_seed"] == 17
            assert stored.initial_context.startswith("The assistant was told")
            assert stored.created_at.tzinfo is not None

            turns = session.scalars(
                select(ConversationTurn)
                .where(ConversationTurn.conversation_id == conversation_id)
                .order_by(ConversationTurn.turn_index)
            ).all()
            assert [(turn.turn_index, turn.role) for turn in turns] == [
                (0, "user"),
                (1, "assistant"),
                (2, "user"),
                (3, "assistant"),
            ]

            session.add(
                ConversationTurn(
                    conversation_id=conversation_id,
                    turn_index=3,
                    role="user",
                    text="This index is already present.",
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()
            session.rollback()

            session.execute(delete(Conversation).where(Conversation.id == conversation_id))
            session.commit()
            assert (
                session.scalar(
                    select(ConversationTurn).where(
                        ConversationTurn.conversation_id == conversation_id
                    )
                )
                is None
            )

        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.downgrade(config, "0001_enable_pgvector")
        remaining_tables = set(inspect(engine).get_table_names())
        assert "conversations" not in remaining_tables
        assert "conversation_turns" not in remaining_tables
    finally:
        engine.dispose()

