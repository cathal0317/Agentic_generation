"""Synthesize one Gemini TTS WAV from a validated JSON input."""

import argparse
import json
import sys
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_generation.tts.gemini import build_synthesis_request, synthesize_wav

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
TIMEOUT_MILLISECONDS = 60_000


class SynthesisInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    vocal_direction: str
    voice: str
    model: str

    @field_validator("text", "vocal_direction", "voice", "model")
    @classmethod
    def require_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class GeminiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        extra="ignore",
        hide_input_in_errors=True,
        str_strip_whitespace=True,
    )

    api_key: SecretStr = Field(min_length=1)


def load_input(path: Path) -> SynthesisInput:
    return SynthesisInput.model_validate(json.loads(path.read_text(encoding="utf-8")))


def validate_output_path(path: Path) -> None:
    if path.suffix.lower() != ".wav":
        raise ValueError("output path must use the .wav extension")
    if path.exists():
        raise FileExistsError(f"output path already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {path.parent}")


def build_dry_run(input_data: SynthesisInput, output_path: Path) -> dict[str, object]:
    return {
        "status": "dry_run",
        "output_path": str(output_path),
        "request": build_synthesis_request(**input_data.model_dump()),
    }


def build_success(result) -> dict[str, object]:
    return {
        "status": "completed",
        "output_path": str(result.output_path),
        "model": result.model,
        "voice": result.voice,
        "audio": {
            "duration_seconds": result.frame_count / result.sample_rate_hz,
            "format": result.audio_format,
            "mime_type": result.mime_type,
            "sample_rate_hz": result.sample_rate_hz,
            "channels": result.channels,
            "sample_width_bits": result.sample_width_bits,
            "frame_count": result.frame_count,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        input_data = load_input(args.input)
        validate_output_path(args.output)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as error:
        print(f"Invalid synthesis input or output: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps(build_dry_run(input_data, args.output), indent=2, ensure_ascii=False))
        return 0

    try:
        settings = GeminiSettings(_env_file=ENV_FILE, _env_file_encoding="utf-8")
    except ValidationError:
        print("GEMINI_API_KEY is missing or invalid.", file=sys.stderr)
        return 2

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"Could not create output directory: {error}", file=sys.stderr)
        return 2

    client = None
    try:
        client = genai.Client(
            api_key=settings.api_key.get_secret_value(),
            http_options=types.HttpOptions(timeout=TIMEOUT_MILLISECONDS),
        )
        result = synthesize_wav(
            client,
            **input_data.model_dump(),
            output_path=args.output,
        )
    except Exception:
        print("Gemini TTS synthesis failed.", file=sys.stderr)
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    print(json.dumps(build_success(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
