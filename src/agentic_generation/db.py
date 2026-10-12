"""Database connection and a small readiness check."""

from dataclasses import dataclass

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase

from agentic_generation.config import Settings


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


def create_db_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


class DatabaseNotReady(RuntimeError):
    """The database is reachable but its schema is not ready."""


@dataclass(frozen=True)
class DatabaseCheck:
    database: str
    postgres_version: str
    pgvector_version: str
    migration: str
    vector_check: str


def check_database(engine: Engine, expected_revision: str) -> DatabaseCheck:
    with engine.connect() as connection:
        database, postgres_version = connection.execute(
            text("SELECT current_database(), current_setting('server_version')")
        ).one()
        vector_version = connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        if vector_version is None:
            raise DatabaseNotReady("pgvector is not enabled. Run: uv run alembic upgrade head")
        if connection.scalar(text("SELECT to_regclass('alembic_version')")) is None:
            raise DatabaseNotReady("Migrations have not run. Run: uv run alembic upgrade head")
        revisions = list(connection.scalars(text("SELECT version_num FROM alembic_version")))
        if revisions != [expected_revision]:
            raise DatabaseNotReady("Schema is not at head. Run: uv run alembic upgrade head")

        # Exercise the extension itself; no permanent table or sample dimension is imposed.
        distance = connection.scalar(text("SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector"))
        if distance is None or abs(float(distance) - 1.0) > 1e-9:
            raise DatabaseNotReady("The pgvector cosine-distance check failed.")

        return DatabaseCheck(
            database=database,
            postgres_version=postgres_version,
            pgvector_version=vector_version,
            migration=expected_revision,
            vector_check="ok",
        )
