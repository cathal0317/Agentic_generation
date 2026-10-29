import base64
import json
import wave
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "synthesize_gemini.py"
SCRIPT_SPEC = spec_from_file_location("synthesize_gemini", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
script = module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(script)


def make_wav_bytes():
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 24)
    return buffer.getvalue()


def valid_input():
    return {
        "text": "They moved the meeting to Friday.",
        "vocal_direction": "Sound quietly pleased.",
        "voice": "Kore",
        "model": "gemini-3.8-flash-tts",
    }


def write_input(tmp_path, data=None):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data or valid_input()), encoding="utf-8")
    return path


class FakeInteractions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.interactions = FakeInteractions(response, error)
        self.closed = False

    def close(self):
        self.closed = True


def audio_response():
    return SimpleNamespace(
        status="completed",
        output_audio=SimpleNamespace(
            data=base64.b64encode(make_wav_bytes()).decode("ascii"),
            mime_type="audio/wav",
        ),
    )


def test_dry_run_needs_no_key_client_or_file_write(tmp_path, monkeypatch, capsys):
    input_path = write_input(tmp_path)
    output_path = tmp_path / "new" / "speech.wav"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fail_client(**_kwargs):
        raise AssertionError("dry-run must not create a client")

    monkeypatch.setattr(script.genai, "Client", fail_client)

    assert script.main(["--input", str(input_path), "--output", str(output_path), "--dry-run"]) == 0

    output = json.loads(capsys.readouterr().out)
    request = output["request"]
    content = request["input"][0]["content"][0]
    assert output["status"] == "dry_run"
    assert request["model"] == "gemini-3.8-flash-tts"
    assert content["text"] == "They moved the meeting to Friday."
    assert content["annotations"] == [
        {"type": "speech_metadata", "style": "Sound quietly pleased."}
    ]
    assert request["response_format"] == {"type": "audio", "mime_type": "audio/wav"}
    assert request["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert not output_path.parent.exists()


@pytest.mark.parametrize(
    "data",
    [
        {**valid_input(), "private_context": "not allowed"},
        {key: value for key, value in valid_input().items() if key != "voice"},
        {**valid_input(), "vocal_direction": " \n "},
        {**valid_input(), "text": 123},
    ],
)
def test_invalid_input_does_not_create_client(tmp_path, monkeypatch, data):
    input_path = write_input(tmp_path, data)

    def fail_client(**_kwargs):
        raise AssertionError("invalid input must not create a client")

    monkeypatch.setattr(script.genai, "Client", fail_client)

    assert script.main(["--input", str(input_path), "--output", str(tmp_path / "out.wav")]) == 2


def test_existing_output_does_not_create_client(tmp_path, monkeypatch):
    input_path = write_input(tmp_path)
    output_path = tmp_path / "out.wav"
    output_path.write_bytes(b"existing")

    def fail_client(**_kwargs):
        raise AssertionError("existing output must not create a client")

    monkeypatch.setattr(script.genai, "Client", fail_client)

    assert script.main(["--input", str(input_path), "--output", str(output_path)]) == 2
    assert output_path.read_bytes() == b"existing"


def test_environment_overrides_dotenv_and_fake_client_succeeds(tmp_path, monkeypatch, capsys):
    input_path = write_input(tmp_path)
    output_path = tmp_path / "results" / "speech.wav"
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEMINI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setattr(script, "ENV_FILE", dotenv)
    monkeypatch.setenv("GEMINI_API_KEY", "environment-key")
    client = FakeClient(audio_response())
    constructor_calls = []

    def fake_client(**kwargs):
        constructor_calls.append(kwargs)
        return client

    monkeypatch.setattr(script.genai, "Client", fake_client)

    assert script.main(["--input", str(input_path), "--output", str(output_path)]) == 0

    output = json.loads(capsys.readouterr().out)
    assert constructor_calls[0]["api_key"] == "environment-key"
    assert constructor_calls[0]["http_options"].timeout == 60_000
    assert client.closed
    assert output["status"] == "completed"
    assert output["output_path"] == str(output_path)
    assert output["model"] == "gemini-3.8-flash-tts"
    assert output["voice"] == "Kore"
    assert output["audio"]["duration_seconds"] == 0.001
    assert output["audio"]["format"] == "wav"
    request_content = client.interactions.calls[0]["input"][0]["content"][0]
    assert request_content["text"] == valid_input()["text"]
    assert request_content["annotations"][0]["style"] == valid_input()["vocal_direction"]
    assert output_path.exists()


def test_missing_key_does_not_create_client(tmp_path, monkeypatch, capsys):
    input_path = write_input(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(script, "ENV_FILE", tmp_path / "missing.env")

    def fail_client(**_kwargs):
        raise AssertionError("missing key must not create a client")

    monkeypatch.setattr(script.genai, "Client", fail_client)

    assert script.main(["--input", str(input_path), "--output", str(tmp_path / "out.wav")]) == 2
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY" in captured.err
    assert captured.out == ""


def test_api_failure_is_not_reported_as_success(tmp_path, monkeypatch, capsys):
    input_path = write_input(tmp_path)
    output_path = tmp_path / "out.wav"
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(script, "ENV_FILE", tmp_path / "missing.env")
    client = FakeClient(error=RuntimeError("provider failed"))
    monkeypatch.setattr(script.genai, "Client", lambda **_kwargs: client)

    assert script.main(["--input", str(input_path), "--output", str(output_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed" in captured.err.lower()
    assert "test-key" not in captured.err
    assert client.closed
    assert not output_path.exists()


def test_examples_match_except_for_vocal_direction():
    wanted = json.loads(
        (script.PROJECT_ROOT / "examples/tts/meeting_wanted.json").read_text(encoding="utf-8")
    )
    unwanted = json.loads(
        (script.PROJECT_ROOT / "examples/tts/meeting_unwanted.json").read_text(encoding="utf-8")
    )

    script.SynthesisInput.model_validate(wanted)
    script.SynthesisInput.model_validate(unwanted)
    assert set(wanted) == {"text", "vocal_direction", "voice", "model"}
    assert set(unwanted) == set(wanted)
    assert wanted["text"] == unwanted["text"]
    assert wanted["voice"] == unwanted["voice"]
    assert wanted["model"] == unwanted["model"] == "gemini-3.8-flash-tts"
    assert wanted["vocal_direction"] != unwanted["vocal_direction"]
