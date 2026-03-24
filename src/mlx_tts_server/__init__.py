# SPDX-License-Identifier: Apache-2.0
"""MLX TTS Server — OpenAI-compatible Text-to-Speech for Apple Silicon."""

from mlx_tts_server.config import TTSConfig, is_tts_model
from mlx_tts_server.protocol import SpeechSynthesisRequest, SpeechSynthesisResult

__all__ = [
    "TTSConfig",
    "SpeechSynthesisRequest",
    "SpeechSynthesisResult",
    "is_tts_model",
]
