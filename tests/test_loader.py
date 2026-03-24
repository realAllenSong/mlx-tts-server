import pytest
from mlx_tts_server.registry import get_tts_model_type, get_tts_model_constructor


def test_get_tts_model_type_qwen():
    assert get_tts_model_type("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice") == "qwen3_tts"
    assert get_tts_model_type("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit") == "qwen3_tts"
    assert get_tts_model_type("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16") == "qwen3_tts"


def test_get_tts_model_type_unknown():
    with pytest.raises(ValueError, match="Cannot determine TTS model type"):
        get_tts_model_type("unknown/model")


def test_get_tts_model_constructor_qwen():
    ctor = get_tts_model_constructor("qwen3_tts")
    assert callable(ctor)


def test_get_tts_model_constructor_unknown():
    with pytest.raises(ValueError, match="Unsupported TTS model_type"):
        get_tts_model_constructor("totally_unknown_type")
