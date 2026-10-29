import base64
import wave
from io import BytesIO
from types import SimpleNamespace

import pytest

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
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response):
        self.interactions = FakeInteractions(response)


def audio_response(data=None, mime_type="audio/wav", status="completed"):
    audio = None
    if data is not None:
        audio = SimpleNamespace(
            data=base64.b64encode(data).decode("ascii"),
            mime_type=mime_type,
        )
    return SimpleNamespace(status=status, output_audio=audio)


def test_synthesizes_verbatim_styled_text_and_saves_playable_wav(tmp_path):
    client = FakeClient(audio_response(make_wav_bytes()))
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
