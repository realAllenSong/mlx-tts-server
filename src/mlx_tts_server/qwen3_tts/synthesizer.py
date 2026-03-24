# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS synthesizer: wraps mlx-audio for generation."""

from __future__ import annotations

import logging
import threading
from typing import Iterator

import numpy as np

from mlx_tts_server.qwen3_tts.config import (
    OPENAI_VOICE_MAP,
    QWEN3_TTS_CUSTOM_VOICES,
    SUPPORTED_LANGUAGES,
    Qwen3TTSConfig,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _detect_language(text: str) -> str:
    """Detect language of text, returning a Qwen3-TTS language code.

    Falls back to 'english' if detection fails or library not installed.
    """
    try:
        from langdetect import detect
        lang = detect(text)
        _LANG_MAP = {
            "zh": "chinese", "zh-cn": "chinese", "zh-tw": "chinese",
            "en": "english",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "es": "spanish",
            "ja": "japanese",
            "ko": "korean",
            "fr": "french",
            "ru": "russian",
        }
        return _LANG_MAP.get(lang.lower(), "english")
    except Exception:
        return "english"


class Qwen3TTSSynthesizer:
    """High-level TTS synthesizer backed by mlx-audio Qwen3-TTS.

    Supports all Qwen3-TTS variants:
    - CustomVoice: preset speakers (ryan, serena, etc.)
    - Base: voice cloning via 3-second reference audio
    - VoiceDesign: voice creation from text description
    """

    def __init__(self, model: object, config: Qwen3TTSConfig) -> None:
        self._model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, model_id: str) -> "Qwen3TTSSynthesizer":
        """Load Qwen3-TTS model from mlx-community or HuggingFace."""
        config = Qwen3TTSConfig.from_model_id(model_id)
        with _MODEL_CACHE_LOCK:
            if model_id in _MODEL_CACHE:
                logger.info("TTS: using cached model for %s", model_id)
                return cls(_MODEL_CACHE[model_id], config)
            logger.info("TTS: loading model %s", model_id)
            from mlx_audio.tts.utils import load_model
            model = load_model(model_id)
            _MODEL_CACHE[model_id] = model
            logger.info("TTS: model %s loaded and cached", model_id)
        return cls(model, config)

    @property
    def voices(self) -> list[str]:
        """Return available voice names (CustomVoice models only)."""
        if self.config.is_custom_voice:
            model_voices = getattr(self._model, "voices", None)
            if model_voices:
                return list(model_voices)
            return list(QWEN3_TTS_CUSTOM_VOICES)
        return []

    @property
    def supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def _resolve_voice(self, voice: str) -> str:
        """Normalize voice: OpenAI aliases -> Qwen3-TTS names (lowercase)."""
        voice_lower = voice.lower()
        if voice_lower in OPENAI_VOICE_MAP:
            return OPENAI_VOICE_MAP[voice_lower]
        known = self.voices
        for v in known:
            if v.lower() == voice_lower:
                return v
        if self.config.is_custom_voice and known:
            logger.warning("TTS: voice %r not recognized, falling back to %r", voice, known[0])
            return known[0]
        return voice_lower

    def _resolve_language(self, language: str | None, text: str) -> str:
        """Resolve language: explicit > auto-detect > 'english'.

        CRITICAL: 'auto' activates NOTHINK mode in the codec — always resolve
        to an explicit language name for proper speech generation.
        """
        if language and language.lower() != "auto":
            lang = language.lower()
            if lang in SUPPORTED_LANGUAGES:
                return lang
            logger.warning("TTS: unknown language %r, falling back to auto-detect", language)
        detected = _detect_language(text)
        logger.debug("TTS: auto-detected language=%r for text %r...", detected, text[:40])
        return detected

    @staticmethod
    def _ensure_wav(path: str) -> str:
        """Convert non-WAV audio to 24kHz mono WAV via ffmpeg if needed."""
        import subprocess
        from pathlib import Path
        p = Path(path)
        with open(p, "rb") as f:
            magic = f.read(4)
        if magic == b"RIFF":
            return path
        import tempfile
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.close()
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(p), "-ar", "24000", "-ac", "1", out.name, "-y"],
                capture_output=True, check=True, timeout=30,
            )
            logger.debug("TTS: converted %s -> %s (24kHz WAV)", p.name, out.name)
            return out.name
        except Exception as e:
            Path(out.name).unlink(missing_ok=True)
            raise ValueError(
                f"ref_audio is not WAV and ffmpeg conversion failed: {e}. "
                "Install ffmpeg: brew install ffmpeg"
            ) from e

    def generate(
        self,
        text: str,
        *,
        voice: str = "ryan",
        language: str | None = None,
        instruct: str | None = None,
        speed: float = 1.0,
        max_tokens: int = 4096,
        temperature: float | None = None,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice: Speaker name (CustomVoice) or ignored (Base/VoiceDesign).
                   Accepts OpenAI aliases: alloy, echo, fable, onyx, nova, shimmer.
            language: Language code. Auto-detected from text if not specified.
                      Supported: english, chinese, german, french, spanish,
                      italian, portuguese, japanese, korean, russian.
            instruct: Emotion/style instruction (CustomVoice) or voice description
                      (VoiceDesign). E.g. "speak slowly and warmly".
            speed: Speech speed multiplier (0.5-2.0).
            temperature: Sampling temperature. None = auto (0.7 for speech, 0.3 for
                         voice cloning). Lower = more faithful to reference voice.
            ref_audio: Path to reference audio file for voice cloning (Base model).
            ref_text: Transcript of reference audio (required with ref_audio).

        Returns:
            (float32_waveform, sample_rate) — sample_rate is 24000 Hz.
        """
        if not text.strip():
            raise ValueError("Input text must not be empty")

        lang = self._resolve_language(language, text)
        is_clone = ref_audio is not None

        if temperature is None:
            temperature = 0.3 if is_clone else 0.7

        import mlx.core as mx

        if is_clone:
            if self.config.is_custom_voice:
                raise ValueError(
                    "Voice cloning (ref_audio) requires a Base model, not CustomVoice. "
                    "Use: mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit"
                )
            if ref_text is None:
                raise ValueError("ref_text is required when ref_audio is provided")
            ref_audio = self._ensure_wav(ref_audio)
            return self._generate_voice_clone(
                text, lang, ref_audio, ref_text,
                max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            )

        if self.config.is_voice_design:
            if instruct is None:
                raise ValueError(
                    "instruct is required for VoiceDesign models — describe the desired voice"
                )
            return self._generate_voice_design(
                text, instruct, lang,
                max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            )

        resolved_voice = self._resolve_voice(voice)
        logger.debug(
            "TTS: CustomVoice synthesis voice=%r lang=%r %d chars",
            resolved_voice, lang, len(text),
        )

        chunks: list[np.ndarray] = []
        sample_rate = self.config.sample_rate

        for result in self._model.generate(
            text,
            voice=resolved_voice,
            instruct=instruct,
            lang_code=lang,
            speed=speed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=False,
        ):
            audio_chunk = result.audio
            if isinstance(audio_chunk, mx.array):
                audio_chunk = np.array(audio_chunk)
            chunks.append(audio_chunk.astype(np.float32))
            if hasattr(result, "sample_rate") and result.sample_rate:
                sample_rate = result.sample_rate

        if not chunks:
            raise RuntimeError("TTS generation produced no audio output")

        return np.concatenate(chunks), sample_rate

    def _generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: str,
        ref_text: str,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """Voice cloning via reference audio (Base model)."""
        logger.debug("TTS: voice cloning from ref_audio=%r lang=%r", ref_audio, language)

        import mlx.core as mx

        chunks: list[np.ndarray] = []
        sample_rate = self.config.sample_rate

        for result in self._model.generate(
            text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            lang_code=language,
            stream=False,
            **kwargs,
        ):
            audio_chunk = result.audio
            if isinstance(audio_chunk, mx.array):
                audio_chunk = np.array(audio_chunk)
            chunks.append(audio_chunk.astype(np.float32))
            if hasattr(result, "sample_rate") and result.sample_rate:
                sample_rate = result.sample_rate

        if not chunks:
            raise RuntimeError("Voice cloning produced no audio output")

        return np.concatenate(chunks), sample_rate

    def _generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str,
        **kwargs,
    ) -> tuple[np.ndarray, int]:
        """Generate speech with voice designed from text description."""
        logger.debug("TTS: VoiceDesign instruct=%r lang=%r", instruct[:60], language)

        import mlx.core as mx

        chunks: list[np.ndarray] = []
        sample_rate = self.config.sample_rate

        for result in self._model.generate(
            text,
            instruct=instruct,
            lang_code=language,
            stream=False,
            **kwargs,
        ):
            audio_chunk = result.audio
            if isinstance(audio_chunk, mx.array):
                audio_chunk = np.array(audio_chunk)
            chunks.append(audio_chunk.astype(np.float32))
            if hasattr(result, "sample_rate") and result.sample_rate:
                sample_rate = result.sample_rate

        if not chunks:
            raise RuntimeError("VoiceDesign generation produced no audio output")

        return np.concatenate(chunks), sample_rate

    def generate_streaming(
        self,
        text: str,
        *,
        voice: str = "ryan",
        language: str | None = None,
        instruct: str | None = None,
        speed: float = 1.0,
        streaming_interval: float = 2.0,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate speech in streaming chunks (for future streaming endpoint)."""
        if not text.strip():
            raise ValueError("Input text must not be empty")

        lang = self._resolve_language(language, text)
        resolved_voice = self._resolve_voice(voice) if self.config.is_custom_voice else voice

        import mlx.core as mx

        kwargs = dict(
            lang_code=lang,
            stream=True,
            streaming_interval=streaming_interval,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if ref_audio is not None:
            gen = self._model.generate(text, ref_audio=ref_audio, ref_text=ref_text, **kwargs)
        elif self.config.is_voice_design:
            gen = self._model.generate(text, instruct=instruct, **kwargs)
        else:
            gen = self._model.generate(text, voice=resolved_voice, instruct=instruct, speed=speed, **kwargs)

        for result in gen:
            audio_chunk = result.audio
            if isinstance(audio_chunk, mx.array):
                audio_chunk = np.array(audio_chunk)
            yield audio_chunk.astype(np.float32)
