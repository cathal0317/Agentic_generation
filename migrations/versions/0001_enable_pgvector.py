"""Enable pgvector in this database.

Revision ID: 0001_enable_pgvector
Revises: none
"""

from alembic import op

revision = "0001_enable_pgvector"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    # The extension may predate this project or be used by other tables.
    # Leave it installed when rolling back our migration history.
    pass
