"""Add conversation and turn storage.

Revision ID: 0002_add_conversations
Revises: 0001_enable_pgvector
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_add_conversations"
down_revision = "0001_enable_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scenario", sa.Text(), nullable=False),
        sa.Column("user_personality", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("initial_condition", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("initial_context", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_table(
        "conversation_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name=op.f("ck_conversation_turns_role_allowed"),
        ),
        sa.CheckConstraint(
            "length(btrim(text)) > 0",
            name=op.f("ck_conversation_turns_text_non_blank"),
        ),
        sa.CheckConstraint(
            "turn_index >= 0",
            name=op.f("ck_conversation_turns_turn_index_non_negative"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_conversation_turns_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_turns")),
        sa.UniqueConstraint(
            "conversation_id",
            "turn_index",
            name=op.f("uq_conversation_turns_conversation_id"),
        ),
    )


def downgrade() -> None:
    op.drop_table("conversation_turns")
    op.drop_table("conversations")
