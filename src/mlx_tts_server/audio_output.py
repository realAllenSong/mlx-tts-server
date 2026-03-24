# SPDX-License-Identifier: Apache-2.0
"""Audio encoding for TTS output: WAV, MP3, FLAC, OPUS, PCM."""

from __future__ import annotations

import io
import subprocess

import numpy as np

MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


def encode_audio(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a float32 waveform to the requested format."""
    fmt = fmt.lower()
    if fmt == "wav":
        return _encode_wav(audio, sample_rate)
    elif fmt == "flac":
        return _encode_soundfile(audio, sample_rate, "FLAC")
    elif fmt == "pcm":
        return _encode_pcm(audio)
    elif fmt in ("mp3", "opus", "aac"):
        return _encode_via_ffmpeg(audio, sample_rate, fmt)
    else:
        raise ValueError(
            f"Unsupported format: {fmt!r}. Use wav, mp3, opus, aac, flac, or pcm."
        )


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _encode_soundfile(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format=fmt)
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    return audio.astype(np.float32).tobytes()


def _encode_via_ffmpeg(audio: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    codec_map = {
        "mp3": ("libmp3lame", "mp3"),
        "opus": ("libopus", "opus"),
        "aac": ("aac", "adts"),
    }
    codec, container = codec_map[fmt]
    raw_pcm = audio.astype(np.float32).tobytes()
    cmd = [
        "ffmpeg", "-y",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-i", "pipe:0",
        "-c:a", codec,
        "-f", container,
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, input=raw_pcm, capture_output=True, timeout=30)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg to use mp3/opus/aac output.\n"
            "  brew install ffmpeg"
        ) from e
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encoding failed (code {result.returncode}):\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return result.stdout
