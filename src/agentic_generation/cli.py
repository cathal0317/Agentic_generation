"""Command-line checks for scenario input and database readiness."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from alembic.config import Config
from alembic.script import ScriptDirectory
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from agentic_generation.config import GenerationSettings, Settings
from agentic_generation.controller import REVIEWER_SYSTEM_PROMPT
from agentic_generation.conversation import ConversationProgress, generate_conversation
from agentic_generation.db import DatabaseNotReady, check_database, create_db_engine
from agentic_generation.openai_adapter import (
    GenerationError,
    create_dialogue_reviewer,
    create_text_generator,
)
from agentic_generation.prompts import build_assistant_system_prompt, build_user_system_prompt
from agentic_generation.spec import ConversationSpec


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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


def generate_from_scenario(
    path: Path,
    premise_path: Path | None = None,
    premise_id: str | None = None,
) -> int:
    spec = load_scenario(path)
    if spec is None:
        return 2

    if (premise_path is None) != (premise_id is None):
        print("--premises and --premise-id must be provided together.", file=sys.stderr)
        return 2

    selection = None
    if premise_path is not None and premise_id is not None:
        from agentic_generation.premises import (
            CONTRACT_VERSION,
            PremiseSelectionError,
            load_premise_selection,
        )

        try:
            selection = load_premise_selection(premise_path, premise_id, spec)
        except PremiseSelectionError as error:
            print(f"Could not select premise: {error}", file=sys.stderr)
            return 2

    try:
        settings = GenerationSettings()
    except ValidationError:
        print(
            "Invalid OpenAI settings. Check OPENAI_* values in .env.",
            file=sys.stderr,
        )
        return 2

    run_id = str(uuid4())
    created_at = datetime.now(UTC).isoformat()
    concern = selection.premise.concern if selection is not None else None
    contract = selection.premise.contract if selection is not None else None
    prompt_hashes = {
        "user": _prompt_hash(build_user_system_prompt(spec, concern, contract)),
        "assistant": _prompt_hash(build_assistant_system_prompt()),
        "reviewer": _prompt_hash(REVIEWER_SYSTEM_PROMPT),
    }
    selected_premise = (
        {
            **selection.premise.model_dump(),
            "generation": {
                "prompt_version": selection.prompt_version,
                "model": selection.model,
                "contract_version": (
                    CONTRACT_VERSION if selection.premise.contract is not None else None
                ),
            },
        }
        if selection is not None
        else None
    )
    base_output = {
        "run": {
            "id": run_id,
            "created_at": created_at,
            "prompt_hashes": prompt_hashes,
        },
        "spec": spec.model_dump(),
        "models": {
            "user": settings.user_model,
            "assistant": settings.assistant_model,
            "reviewer": settings.assistant_model,
        },
        "generation_settings": {
            "max_output_tokens": settings.max_output_tokens,
            "timeout_seconds": 60.0,
            "max_retries": 0,
        },
        "selected_premise": selected_premise,
    }
    progress = ConversationProgress()

    def print_failure(reason: str) -> None:
        print(
            json.dumps(
                {
                    **base_output,
                    "status": "failed",
                    "failure": {
                        "stage": progress.current_stage,
                        "reason": reason,
                    },
                    "turns": [asdict(turn) for turn in progress.turns],
                    "termination_reason": None,
                    "control_trace": [asdict(record) for record in progress.control_trace],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

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
                stage="user",
            ),
            create_text_generator(
                client,
                settings.assistant_model,
                settings.max_output_tokens,
                stage="assistant",
            ),
            create_dialogue_reviewer(
                client,
                settings.assistant_model,
                settings.max_output_tokens,
            ),
            opening=selection.premise.opening if selection is not None else None,
            concern=selection.premise.concern if selection is not None else None,
            contract=selection.premise.contract if selection is not None else None,
            progress=progress,
        )
    except GenerationError as error:
        print_failure(str(error))
        print(
            f"Conversation generation failed: {error} (run_id={run_id})",
            file=sys.stderr,
        )
        return 1
    except APITimeoutError:
        print_failure("openai_timeout")
        print(
            f"Conversation generation failed: OpenAI request timed out. (run_id={run_id})",
            file=sys.stderr,
        )
        return 1
    except APIConnectionError:
        print_failure("openai_connection_error")
        print(
            f"Conversation generation failed: could not connect to OpenAI. (run_id={run_id})",
            file=sys.stderr,
        )
        return 1
    except APIStatusError as error:
        print_failure(f"openai_http_{error.status_code}")
        print(
            "Conversation generation failed: "
            f"OpenAI returned HTTP {error.status_code}. (run_id={run_id})",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print_failure("unexpected_error")
        print(
            "Conversation generation failed. Check OpenAI configuration and availability. "
            f"(run_id={run_id})",
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
                **base_output,
                "status": "completed",
                "failure": None,
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
    generate_parser.add_argument("--premises", type=Path)
    generate_parser.add_argument("--premise-id")
    args = parser.parse_args()

    if args.command == "validate-scenario":
        return validate_scenario(args.path)
    if args.command == "preview-prompts":
        return preview_prompts(args.path)
    if args.command == "generate":
        return generate_from_scenario(args.path, args.premises, args.premise_id)

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
