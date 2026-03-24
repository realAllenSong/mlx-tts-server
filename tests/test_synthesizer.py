import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mlx_tts_server.qwen3_tts.config import (
    QWEN3_TTS_CUSTOM_VOICES, OPENAI_VOICE_MAP, Qwen3TTSConfig
)


def test_qwen3_tts_config_custom_voice():
    cfg = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    assert cfg.variant == "CustomVoice"
    assert cfg.size == "0.6B"
    assert cfg.is_custom_voice is True
    assert cfg.sample_rate == 24000


def test_qwen3_tts_config_base():
    cfg = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    assert cfg.variant == "Base"
    assert cfg.size == "1.7B"
    assert cfg.is_base is True


def test_voice_list_not_empty():
    assert len(QWEN3_TTS_CUSTOM_VOICES) > 0
    assert "ryan" in QWEN3_TTS_CUSTOM_VOICES


def test_openai_voice_map():
    assert "alloy" in OPENAI_VOICE_MAP
    assert "echo" in OPENAI_VOICE_MAP
    assert OPENAI_VOICE_MAP["echo"] == "ryan"


def test_synthesizer_resolve_voice_openai_alias():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    mock_model.voices = QWEN3_TTS_CUSTOM_VOICES
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    assert synth._resolve_voice("echo") == "ryan"
    assert synth._resolve_voice("alloy") == "serena"
    assert synth._resolve_voice("ryan") == "ryan"


def test_synthesizer_resolve_voice_case_insensitive():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    mock_model.voices = QWEN3_TTS_CUSTOM_VOICES
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    assert synth._resolve_voice("RYAN") == "ryan"


def test_synthesizer_generate_empty_raises():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    mock_model.voices = QWEN3_TTS_CUSTOM_VOICES
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    with pytest.raises(ValueError, match="empty"):
        synth.generate("   ", voice="ryan")


def test_synthesizer_model_caching():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer, _MODEL_CACHE
    import mlx_tts_server.qwen3_tts.synthesizer as synth_module

    fake_model = MagicMock()
    fake_model.voices = ["ryan"]

    with patch("mlx_audio.tts.utils.load_model", return_value=fake_model) as mock_load:
        original_cache = synth_module._MODEL_CACHE.copy()
        synth_module._MODEL_CACHE.clear()
        try:
            synth1 = Qwen3TTSSynthesizer.from_pretrained("mlx-community/Qwen3-TTS-test-model")
            synth2 = Qwen3TTSSynthesizer.from_pretrained("mlx-community/Qwen3-TTS-test-model")
            assert mock_load.call_count == 1
            assert synth1._model is synth2._model
        finally:
            synth_module._MODEL_CACHE.clear()
            synth_module._MODEL_CACHE.update(original_cache)


def test_resolve_language_auto_detect_english():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    mock_model.voices = QWEN3_TTS_CUSTOM_VOICES
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    lang = synth._resolve_language(None, "The quick brown fox jumps over the lazy dog.")
    assert lang != "auto"
    assert lang == "english"


def test_resolve_language_explicit():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    mock_model.voices = QWEN3_TTS_CUSTOM_VOICES
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    assert synth._resolve_language("french", "Bonjour") == "french"
    assert synth._resolve_language("japanese", "こんにちは") == "japanese"


def test_resolve_language_rejects_auto():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    lang = synth._resolve_language("auto", "Hello world")
    assert lang != "auto"


def test_voice_clone_requires_ref_text():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    with pytest.raises(ValueError, match="ref_text is required"):
        synth.generate("Hello.", ref_audio="ref.wav")


def test_voice_design_requires_instruct():
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    mock_model = MagicMock()
    config = Qwen3TTSConfig.from_model_id("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    synth = Qwen3TTSSynthesizer(mock_model, config)
    with pytest.raises(ValueError, match="instruct is required"):
        synth.generate("Hello.")


def test_supported_languages():
    from mlx_tts_server.qwen3_tts.config import SUPPORTED_LANGUAGES
    assert "english" in SUPPORTED_LANGUAGES
    assert "chinese" in SUPPORTED_LANGUAGES
    assert len(SUPPORTED_LANGUAGES) >= 10


@pytest.mark.slow
def test_synthesizer_real_generation():
    """Integration test: English speech must have non-trivial RMS (not breathing)."""
    from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer
    MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    synth = Qwen3TTSSynthesizer.from_pretrained(MODEL_ID)

    assert synth.voices
    assert "ryan" in synth.voices

    audio, sr = synth.generate(
        "Hello from Apple Silicon! This is Qwen3-TTS working correctly.",
        voice="ryan",
    )
    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert len(audio) > 0
    assert audio.dtype == np.float32
    rms = float(np.sqrt(np.mean(audio ** 2)))
    assert rms > 0.02, f"Audio sounds like breathing (RMS={rms:.4f}), NOTHINK bug still present"
