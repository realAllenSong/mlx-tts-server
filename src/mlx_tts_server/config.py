# SPDX-License-Identifier: Apache-2.0
"""Configuration for Text-to-Speech."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_TTS_MODEL_TYPES = frozenset({"qwen3_tts", "qwen3tts"})

_TTS_NAME_PATTERNS = (
    "qwen3-tts",
    "qwen3_tts",
)


@dataclass
class TTSConfig:
    """Runtime configuration for TTS synthesis."""
    sample_rate: int = 24000
    default_format: str = "wav"
    default_voice: str = "ryan"
    max_tokens: int = 4096
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    do_sample: bool = True
    streaming_interval: float = 2.0


def is_tts_model(model_path: str) -> bool:
    """Return True if model_path points to a TTS model."""
    path_lower = model_path.lower().replace("_", "-")
    if any(p.lower().replace("_", "-") in path_lower for p in _TTS_NAME_PATTERNS):
        return True

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            model_type = cfg.get("model_type", "").lower().replace("_", "")
            return model_type in _TTS_MODEL_TYPES
        except Exception as e:
            logger.debug("Could not read TTS config at %s: %s", config_path, e)

    return False
