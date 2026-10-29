import base64
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest

import agentic_generation.tts.gemini as gemini
from agentic_generation.tts.gemini import GeminiTtsError, synthesize_wav


def make_wav_bytes():
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x00" * 24)
    return buffer.getvalue()


class FakeInteractions:
    def __init__(self, response, on_create=None):
        self.response = response
        self.on_create = on_create
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.on_create is not None:
            self.on_create()
        return self.response


class FakeClient:
    def __init__(self, response, on_create=None):
        self.interactions = FakeInteractions(response, on_create)


def audio_response(data=None, mime_type="audio/wav", status="completed"):
    audio = None
    if data is not None:
        audio = SimpleNamespace(
            data=base64.b64encode(data).decode("ascii"),
            mime_type=mime_type,
        )
    return SimpleNamespace(status=status, output_audio=audio)


def test_synthesizes_verbatim_styled_text_and_saves_playable_wav(tmp_path):
    wav_bytes = make_wav_bytes()
    client = FakeClient(audio_response(wav_bytes))
    output_path = tmp_path / "speech.wav"

    result = synthesize_wav(
        client,
        text="Keep this text, exactly!",
        vocal_direction="Calm, measured delivery.",
        voice="Kore",
        model="gemini-3.8-flash-tts",
        output_path=output_path,
    )

    assert len(client.interactions.calls) == 1
    assert client.interactions.calls[0] == {
        "model": "gemini-3.8-flash-tts",
        "input": [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": "Keep this text, exactly!",
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": "Calm, measured delivery.",
                            }
                        ],
                    }
                ],
            }
        ],
        "response_format": {"type": "audio", "mime_type": "audio/wav"},
        "generation_config": {"speech_config": [{"voice": "Kore"}]},
    }
    with wave.open(str(output_path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        assert wav.getnframes() == 24
    assert output_path.read_bytes() == wav_bytes
    assert result.output_path == output_path
    assert result.model == "gemini-3.8-flash-tts"
    assert result.voice == "Kore"
    assert result.mime_type == "audio/wav"
    assert result.audio_format == "wav"
    assert result.sample_rate_hz == 24000
    assert result.channels == 1
    assert result.sample_width_bits == 16
    assert result.frame_count == 24


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", ""),
        ("text", " \n "),
        ("vocal_direction", ""),
        ("vocal_direction", "   "),
    ],
)
def test_rejects_empty_input_without_calling_api(tmp_path, field, value):
    client = FakeClient(audio_response(make_wav_bytes()))
    arguments = {
        "text": "Hello.",
        "vocal_direction": "Neutral.",
        "voice": "Kore",
        "model": "gemini-3.8-flash-tts",
        "output_path": tmp_path / "speech.wav",
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=field):
        synthesize_wav(client, **arguments)

    assert client.interactions.calls == []


def test_rejects_existing_output_without_calling_api(tmp_path):
    output_path = tmp_path / "speech.wav"
    output_path.write_bytes(b"existing")
    client = FakeClient(audio_response(make_wav_bytes()))

    with pytest.raises(FileExistsError):
        synthesize_wav(
            client,
            text="Hello.",
            vocal_direction="Neutral.",
            voice="Kore",
            model="gemini-3.8-flash-tts",
            output_path=output_path,
        )

    assert client.interactions.calls == []
    assert output_path.read_bytes() == b"existing"


def test_preserves_destination_created_while_api_call_is_in_flight(tmp_path):
    output_path = tmp_path / "speech.wav"
    client = FakeClient(
        audio_response(make_wav_bytes()),
        on_create=lambda: output_path.write_bytes(b"created by another process"),
    )

    with pytest.raises(FileExistsError):
        synthesize_wav(
            client,
            text="Hello.",
            vocal_direction="Neutral.",
            voice="Kore",
            model="gemini-3.8-flash-tts",
            output_path=output_path,
        )

    assert output_path.read_bytes() == b"created by another process"
    assert list(tmp_path.glob(".speech.wav.*.tmp")) == []


@pytest.mark.parametrize("failure_point", ["write", "publish"])
def test_cleans_only_temporary_file_when_storage_fails(tmp_path, monkeypatch, failure_point):
    output_path = tmp_path / "speech.wav"
    client = FakeClient(audio_response(make_wav_bytes()))

    def fail(*_args):
        raise OSError(f"{failure_point} failed")

    monkeypatch.setattr(
        gemini.os,
        "write" if failure_point == "write" else "link",
        fail,
    )
    with pytest.raises(OSError, match=failure_point):
        synthesize_wav(
            client,
            text="Hello.",
            vocal_direction="Neutral.",
            voice="Kore",
            model="gemini-3.8-flash-tts",
            output_path=output_path,
        )

    assert not output_path.exists()
    assert list(tmp_path.glob(".speech.wav.*.tmp")) == []


def test_rejects_truncated_wav_frame_data_without_file(tmp_path):
    output_path = tmp_path / "speech.wav"
    client = FakeClient(audio_response(make_wav_bytes()[:-2]))

    with pytest.raises(GeminiTtsError, match="truncated"):
        synthesize_wav(
            client,
            text="Hello.",
            vocal_direction="Neutral.",
            voice="Kore",
            model="gemini-3.8-flash-tts",
            output_path=output_path,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    "response",
    [
        audio_response(),
        audio_response(b"\x00\x01", mime_type="audio/l16"),
        audio_response(b"not a wave file"),
        audio_response(make_wav_bytes(), status="failed"),
    ],
)
def test_rejects_missing_unsupported_or_malformed_audio_without_file(tmp_path, response):
    output_path = tmp_path / "speech.wav"
    client = FakeClient(response)

    with pytest.raises(GeminiTtsError):
        synthesize_wav(
            client,
            text="Hello.",
            vocal_direction="Neutral.",
            voice="Kore",
            model="gemini-3.8-flash-tts",
            output_path=output_path,
        )

    assert len(client.interactions.calls) == 1
    assert not output_path.exists()
