# SPDX-License-Identifier: Apache-2.0
"""TTS model loader."""

from __future__ import annotations
import logging
from pathlib import Path
from mlx_tts_server.registry import get_tts_model_constructor, get_tts_model_type

logger = logging.getLogger(__name__)


def load_tts_model(model_path: str | Path) -> object:
    """Load a TTS model from path or HuggingFace repo ID."""
    model_path = str(model_path)
    model_type = get_tts_model_type(model_path)
    logger.info("TTS loader: detected model_type=%s for %s", model_type, model_path)
    constructor = get_tts_model_constructor(model_type)
    return constructor(model_path)
