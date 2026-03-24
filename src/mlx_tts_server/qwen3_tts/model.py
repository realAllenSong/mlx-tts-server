# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model entry point."""
from __future__ import annotations

from mlx_tts_server.qwen3_tts.synthesizer import Qwen3TTSSynthesizer

def load_qwen3_tts_model(model_path: str) -> Qwen3TTSSynthesizer:
    return Qwen3TTSSynthesizer.from_pretrained(model_path)
