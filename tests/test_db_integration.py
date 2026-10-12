"""Opt-in check against a disposable database, never a production database."""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from agentic_generation.db import check_database

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_migrations_and_vector_query_against_postgres():
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL + pgvector database")

    engine = create_engine(database_url, connect_args={"connect_timeout": 5})
    config = Config(str(ROOT / "alembic.ini"))
    try:
        for _ in range(2):
            with engine.begin() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
        expected_revision = ScriptDirectory.from_config(config).get_current_head()
        assert expected_revision is not None
        assert check_database(engine, expected_revision).vector_check == "ok"

        with engine.begin() as connection:
            connection.execute(
                text("CREATE TEMP TABLE vector_probe (id integer, value vector(3)) ON COMMIT DROP")
            )
            connection.execute(
                text("INSERT INTO vector_probe VALUES (1, '[1,0,0]'), (2, '[0,1,0]')")
            )
            nearest = connection.scalar(
                text("SELECT id FROM vector_probe ORDER BY value <=> '[1,0,0]'::vector LIMIT 1")
            )
            assert nearest == 1
    finally:
        engine.dispose()
