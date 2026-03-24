import pytest
from mlx_tts_server.config import TTSConfig, is_tts_model
from mlx_tts_server.protocol import SpeechSynthesisRequest, SpeechSynthesisResult
import numpy as np


def test_is_tts_model_qwen3():
    assert is_tts_model("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice") is True
    assert is_tts_model("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit") is True
    assert is_tts_model("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16") is True


def test_is_tts_model_false():
    assert is_tts_model("Qwen/Qwen3-0.6B") is False
    assert is_tts_model("openai/whisper-small") is False
    assert is_tts_model("Qwen/Qwen3-ASR-0.6B") is False


def test_tts_config_defaults():
    cfg = TTSConfig()
    assert cfg.sample_rate == 24000
    assert cfg.default_format == "wav"
    assert cfg.default_voice == "ryan"
    assert cfg.max_tokens == 4096


def test_speech_synthesis_request_defaults():
    req = SpeechSynthesisRequest(model="tts-1", input="Hello", voice="ryan")
    assert req.response_format == "wav"
    assert req.speed == 1.0


def test_speech_synthesis_request_invalid_model():
    with pytest.raises(Exception):
        SpeechSynthesisRequest(model="", input="Hello", voice="ryan")


def test_speech_synthesis_request_invalid_format():
    with pytest.raises(Exception):
        SpeechSynthesisRequest(model="m", input="Hello", voice="ryan", response_format="xyz")


def test_speech_synthesis_request_invalid_speed():
    with pytest.raises(Exception):
        SpeechSynthesisRequest(model="m", input="Hello", voice="ryan", speed=10.0)


def test_speech_synthesis_result():
    audio = np.zeros(24000, dtype=np.float32)
    result = SpeechSynthesisResult(audio=audio, sample_rate=24000, duration=1.0)
    assert result.duration == 1.0
    assert result.sample_rate == 24000
