from io import StringIO
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]


def configuration(output: StringIO) -> Config:
    config = Config(str(ROOT / "alembic.ini"), output_buffer=output)
    config.attributes["database_url"] = URL.create(
        "postgresql+psycopg", username="test", password="special%password@", database="test"
    )
    return config


def test_upgrade_sql_enables_vector_and_records_revision_without_connecting():
    output = StringIO()
    command.upgrade(configuration(output), "head", sql=True)
    sql = output.getvalue()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "0001_enable_pgvector" in sql
    assert "special%password" not in sql


def test_downgrade_does_not_drop_a_potentially_shared_extension():
    output = StringIO()
    command.downgrade(configuration(output), "head:base", sql=True)
    sql = output.getvalue()
    assert "DROP EXTENSION" not in sql
    assert "DELETE FROM alembic_version" in sql
    assert "0001_enable_pgvector" in sql
