# SPDX-License-Identifier: Apache-2.0
"""TTS server: FastAPI application for OpenAI-compatible speech synthesis."""

from mlx_tts_server.server.app import create_app

__all__ = ["create_app"]
