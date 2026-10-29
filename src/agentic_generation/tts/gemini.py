"""Generate one styled utterance as a WAV file with Gemini TTS."""

import base64
import binascii
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Protocol


class GeminiTtsError(RuntimeError):
    """Gemini returned no usable WAV audio."""


class _Interactions(Protocol):
    def create(self, **kwargs): ...


class GeminiTtsClient(Protocol):
    interactions: _Interactions


@dataclass(frozen=True)
class GeminiTtsResult:
    output_path: Path
    model: str
    voice: str
    mime_type: str
    audio_format: str
    sample_rate_hz: int
    channels: int
    sample_width_bits: int
    frame_count: int


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _decode_wav(audio) -> tuple[bytes, str, int, int, int, int]:
    if audio is None or not getattr(audio, "data", None):
        raise GeminiTtsError("Gemini TTS returned no audio data")
    mime_type = getattr(audio, "mime_type", None)
    if not isinstance(mime_type, str) or mime_type.split(";", 1)[0].lower() != "audio/wav":
        raise GeminiTtsError(f"Gemini TTS returned unsupported audio format: {mime_type!r}")
    data = audio.data
    try:
        wav_bytes = base64.b64decode(data, validate=True) if isinstance(data, str) else bytes(data)
    except (ValueError, TypeError, binascii.Error) as error:
        raise GeminiTtsError("Gemini TTS returned invalid audio encoding") from error
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as wav:
            channels = wav.getnchannels()
            sample_width_bits = wav.getsampwidth() * 8
            sample_rate_hz = wav.getframerate()
            frame_count = wav.getnframes()
            if channels < 1 or sample_width_bits < 1 or sample_rate_hz < 1 or frame_count < 1:
                raise GeminiTtsError("Gemini TTS returned empty or invalid WAV audio")
    except (EOFError, wave.Error) as error:
        raise GeminiTtsError("Gemini TTS returned malformed WAV audio") from error
    return wav_bytes, mime_type, sample_rate_hz, channels, sample_width_bits, frame_count


def synthesize_wav(
    client: GeminiTtsClient,
    *,
    text: str,
    vocal_direction: str,
    voice: str,
    model: str,
    output_path: str | Path,
) -> GeminiTtsResult:
    """Synthesize one verbatim utterance and atomically save a complete WAV."""
    text = _required(text, "text")
    vocal_direction = _required(vocal_direction, "vocal_direction")
    voice = _required(voice, "voice")
    model = _required(model, "model")
    path = Path(output_path)
    if path.suffix.lower() != ".wav":
        raise ValueError("output_path must use the .wav extension")
    if path.exists():
        raise FileExistsError(f"output_path already exists: {path}")
    if not path.parent.exists():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")

    interaction = client.interactions.create(
        model=model,
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": vocal_direction,
                            }
                        ],
                    }
                ],
            }
        ],
        response_format={"type": "audio", "mime_type": "audio/wav"},
        generation_config={"speech_config": [{"voice": voice}]},
    )
    if getattr(interaction, "status", "completed") != "completed":
        raise GeminiTtsError("Gemini TTS interaction did not complete")
    wav_bytes, mime_type, rate, channels, width, frames = _decode_wav(
        getattr(interaction, "output_audio", None)
    )
    try:
        with path.open("xb") as file:
            file.write(wav_bytes)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return GeminiTtsResult(
        output_path=path,
        model=model,
        voice=voice,
        mime_type=mime_type,
        audio_format="wav",
        sample_rate_hz=rate,
        channels=channels,
        sample_width_bits=width,
        frame_count=frames,
    )
