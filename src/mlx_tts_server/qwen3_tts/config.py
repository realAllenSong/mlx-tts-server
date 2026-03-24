# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model configuration."""

from __future__ import annotations
from dataclasses import dataclass

QWEN3_TTS_CUSTOM_VOICES = [
    "ryan", "aiden", "eric", "dylan",
    "serena", "vivian", "sohee",
    "uncle_fu", "ono_anna",
]

SUPPORTED_LANGUAGES = frozenset({
    "english", "chinese", "german", "italian", "portuguese",
    "spanish", "japanese", "korean", "french", "russian",
})

OPENAI_VOICE_MAP = {
    "alloy": "serena",
    "echo": "ryan",
    "fable": "vivian",
    "onyx": "eric",
    "nova": "aiden",
    "shimmer": "sohee",
}


@dataclass
class Qwen3TTSConfig:
    """Configuration for a loaded Qwen3-TTS model."""
    model_id: str
    variant: str    # "CustomVoice" | "Base" | "VoiceDesign"
    size: str       # "0.6B" | "1.7B"
    sample_rate: int = 24000

    @property
    def is_custom_voice(self) -> bool:
        return "customvoice" in self.variant.lower()

    @property
    def is_base(self) -> bool:
        return "base" in self.variant.lower()

    @property
    def is_voice_design(self) -> bool:
        return "voicedesign" in self.variant.lower()

    @classmethod
    def from_model_id(cls, model_id: str) -> "Qwen3TTSConfig":
        mid_lower = model_id.lower()
        if "customvoice" in mid_lower:
            variant = "CustomVoice"
        elif "voicedesign" in mid_lower:
            variant = "VoiceDesign"
        elif "base" in mid_lower:
            variant = "Base"
        else:
            variant = "CustomVoice"

        if "1.7b" in mid_lower:
            size = "1.7B"
        elif "0.6b" in mid_lower:
            size = "0.6B"
        else:
            size = "unknown"

        return cls(model_id=model_id, variant=variant, size=size)
