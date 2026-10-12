"""Alembic uses the same connection settings as the application."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

import agentic_generation.models  # noqa: F401
from agentic_generation.config import Settings
from agentic_generation.db import Base, create_db_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_offline() -> None:
    # Attributes also let tests supply a URL without setting real credentials.
    url = config.attributes.get("database_url")
    if url is None:
        url = Settings().database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_with_connection(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    supplied_connection = config.attributes.get("connection")
    if supplied_connection is not None:
        run_with_connection(supplied_connection)
        return

    engine = create_db_engine(Settings())
    try:
        with engine.connect() as connection:
            run_with_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
