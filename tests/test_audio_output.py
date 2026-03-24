import numpy as np
import pytest
from mlx_tts_server.audio_output import encode_audio, MIME_TYPES

SAMPLE_AUDIO = np.zeros(24000, dtype=np.float32)


def test_encode_wav():
    result = encode_audio(SAMPLE_AUDIO, 24000, "wav")
    assert isinstance(result, bytes)
    assert result[:4] == b"RIFF"


def test_encode_flac():
    result = encode_audio(SAMPLE_AUDIO, 24000, "flac")
    assert isinstance(result, bytes)
    assert result[:4] == b"fLaC"


def test_encode_pcm():
    result = encode_audio(SAMPLE_AUDIO, 24000, "pcm")
    assert isinstance(result, bytes)
    assert len(result) == 24000 * 4


def test_mime_types():
    assert MIME_TYPES["wav"] == "audio/wav"
    assert MIME_TYPES["mp3"] == "audio/mpeg"
    assert MIME_TYPES["pcm"] == "audio/pcm"
    assert MIME_TYPES["flac"] == "audio/flac"


def test_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        encode_audio(SAMPLE_AUDIO, 24000, "xyz")


def test_encode_mp3_requires_ffmpeg():
    """MP3 encoding requires ffmpeg — skip if not installed."""
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    result = encode_audio(SAMPLE_AUDIO, 24000, "mp3")
    assert isinstance(result, bytes)
    assert len(result) > 0
