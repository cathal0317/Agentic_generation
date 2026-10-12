"""Run from the repository root: agentic-generation check-db."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from agentic_generation.config import Settings
from agentic_generation.db import DatabaseNotReady, check_database, create_db_engine


def main() -> int:
    parser = argparse.ArgumentParser(prog="agentic-generation")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-db", help="Check connectivity, migrations, and pgvector")
    parser.parse_args()

    if not Path("alembic.ini").is_file():
        print("Run this command from the repository root containing alembic.ini.", file=sys.stderr)
        return 2
    try:
        settings = Settings()
    except ValidationError:
        print("Invalid database settings. Check .env against .env.example.", file=sys.stderr)
        return 2

    revision = ScriptDirectory.from_config(Config("alembic.ini")).get_current_head()
    if revision is None:
        print("No database migration found.", file=sys.stderr)
        return 2

    engine = create_db_engine(settings)
    try:
        result = check_database(engine, revision)
    except DatabaseNotReady as error:
        print(str(error), file=sys.stderr)
        return 1
    except SQLAlchemyError:
        # Connection errors can contain credentials; report a safe, actionable message.
        print(
            "Database check failed. Check .env and run: docker compose up -d --wait db",
            file=sys.stderr,
        )
        return 1
    finally:
        engine.dispose()

    print(json.dumps(asdict(result), indent=2))
    return 0
