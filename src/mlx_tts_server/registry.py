# SPDX-License-Identifier: Apache-2.0
"""TTS model constructor registry."""

from __future__ import annotations
from typing import Callable

TTSModelConstructor = Callable[[str], object]

_TTS_NAME_TO_TYPE: list[tuple[str, str]] = [
    ("qwen3-tts", "qwen3_tts"),
    ("qwen3_tts", "qwen3_tts"),
]


def _get_qwen3_tts_constructor() -> TTSModelConstructor:
    from mlx_tts_server.qwen3_tts.model import load_qwen3_tts_model
    return load_qwen3_tts_model


def get_tts_model_type(model_path: str) -> str:
    path_lower = model_path.lower()
    for pattern, model_type in _TTS_NAME_TO_TYPE:
        if pattern in path_lower:
            return model_type
    raise ValueError(f"Cannot determine TTS model type from: {model_path!r}")


def get_tts_model_constructor(model_type: str) -> TTSModelConstructor:
    constructors = {
        "qwen3_tts": _get_qwen3_tts_constructor,
        "qwen3tts": _get_qwen3_tts_constructor,
    }
    factory = constructors.get(model_type.lower())
    if factory is None:
        raise ValueError(f"Unsupported TTS model_type: {model_type!r}")
    return factory()
