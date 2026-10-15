"""Command-line checks for scenario input and database readiness."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from openai import OpenAI
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from agentic_generation.config import GenerationSettings, Settings
from agentic_generation.conversation import generate_conversation
from agentic_generation.db import DatabaseNotReady, check_database, create_db_engine
from agentic_generation.openai_adapter import create_dialogue_reviewer, create_text_generator
from agentic_generation.prompts import build_assistant_system_prompt, build_user_system_prompt
from agentic_generation.spec import ConversationSpec


def load_scenario(path: Path) -> ConversationSpec | None:
    try:
        raw_input = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        print(f"Could not read scenario file '{path}': {error}", file=sys.stderr)
        return None

    try:
        data = json.loads(raw_input)
    except json.JSONDecodeError as error:
        print(
            f"Invalid JSON in scenario file '{path}' at line {error.lineno}, column {error.colno}.",
            file=sys.stderr,
        )
        return None

    try:
        return ConversationSpec.model_validate(data)
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in detail["loc"]) or "input"
        print(f"Invalid scenario input at '{location}': {detail['msg']}", file=sys.stderr)
        return None


def validate_scenario(path: Path) -> int:
    if load_scenario(path) is None:
        return 2

    print(f"Scenario is valid: {path}")
    return 0


def preview_prompts(path: Path) -> int:
    spec = load_scenario(path)
    if spec is None:
        return 2

    print(
        json.dumps(
            {
                "user_system_prompt": build_user_system_prompt(spec),
                "assistant_system_prompt": build_assistant_system_prompt(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def generate_from_scenario(path: Path) -> int:
    spec = load_scenario(path)
    if spec is None:
        return 2

    try:
        settings = GenerationSettings()
    except ValidationError:
        print(
            "Invalid OpenAI settings. Check OPENAI_* values in .env.",
            file=sys.stderr,
        )
        return 2

    client = None
    try:
        client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=60.0,
            max_retries=0,
        )
        result = generate_conversation(
            spec,
            create_text_generator(
                client,
                settings.user_model,
                settings.max_output_tokens,
            ),
            create_text_generator(
                client,
                settings.assistant_model,
                settings.max_output_tokens,
            ),
            create_dialogue_reviewer(
                client,
                settings.assistant_model,
                settings.max_output_tokens,
            ),
        )
    except Exception:
        print(
            "Conversation generation failed. Check OpenAI configuration and availability.",
            file=sys.stderr,
        )
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    print(
        json.dumps(
            {
                "spec": spec.model_dump(),
                "models": {
                    "user": settings.user_model,
                    "assistant": settings.assistant_model,
                    "reviewer": settings.assistant_model,
                },
                "turns": [asdict(turn) for turn in result.turns],
                "termination_reason": result.termination_reason,
                "control_trace": [asdict(record) for record in result.control_trace],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agentic-generation")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-db", help="Check connectivity, migrations, and pgvector")
    validate_parser = subcommands.add_parser(
        "validate-scenario", help="Validate a conversation scenario JSON file"
    )
    validate_parser.add_argument("path", type=Path)
    preview_parser = subcommands.add_parser(
        "preview-prompts", help="Preview user and assistant system prompts"
    )
    preview_parser.add_argument("path", type=Path)
    generate_parser = subcommands.add_parser("generate", help="Generate a conversation with OpenAI")
    generate_parser.add_argument("path", type=Path)
    args = parser.parse_args()

    if args.command == "validate-scenario":
        return validate_scenario(args.path)
    if args.command == "preview-prompts":
        return preview_prompts(args.path)
    if args.command == "generate":
        return generate_from_scenario(args.path)

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
