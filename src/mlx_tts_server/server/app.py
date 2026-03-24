# SPDX-License-Identifier: Apache-2.0
"""FastAPI TTS server — OpenAI-compatible /v1/audio/speech."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from mlx_tts_server.protocol import SpeechSynthesisRequest
from mlx_tts_server.server.handlers import handle_speech_synthesis, handle_voice_clone

logger = logging.getLogger(__name__)

_START_TIME = time.time()


def create_app(model_id: str, synthesizer: object) -> FastAPI:
    """Create the FastAPI TTS application.

    Args:
        model_id: HuggingFace model ID or local path.
        synthesizer: Qwen3TTSSynthesizer instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("TTS server: warming up model %s...", model_id)
        try:
            voices = getattr(synthesizer, "voices", ["ryan"])
            warmup_voice = voices[0] if voices else "ryan"
            synthesizer.generate("Hi.", voice=warmup_voice)
            logger.info("TTS server: warm-up complete")
        except Exception as e:
            logger.warning("TTS server: warm-up failed (proceeding anyway): %s", e)
        yield
        logger.info("TTS server: shutting down")

    app = FastAPI(
        title="MLX TTS Server",
        description="OpenAI-compatible Text-to-Speech API for Apple Silicon",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.synthesizer = synthesizer
    app.state.model_id = model_id

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": model_id,
            "uptime": round(time.time() - _START_TIME, 2),
        }

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(_START_TIME),
                    "owned_by": "mlx-tts-server",
                    "capabilities": {"audio_generation": True},
                }
            ],
        }

    @app.get("/v1/audio/speech/voices")
    async def list_voices() -> dict[str, Any]:
        voices = getattr(app.state.synthesizer, "voices", [])
        return {"voices": voices, "model": model_id}

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechSynthesisRequest) -> Response:
        """OpenAI-compatible speech synthesis endpoint."""
        return await handle_speech_synthesis(request, app.state.synthesizer)

    @app.post("/v1/audio/clone")
    async def voice_clone(
        input: str = Form(..., description="Text to synthesize"),
        ref_audio: UploadFile = File(..., description="Reference audio file (~3s WAV/MP3)"),
        ref_text: str = Form("", description="Transcript of reference audio (optional but improves quality)"),
        response_format: str = Form("wav", description="Output format: wav, mp3, flac, opus, aac, pcm"),
        language: str = Form("english", description="Language of the input text"),
        model: str = Form("clone", description="Model identifier (informational)"),
    ) -> Response:
        """Voice cloning endpoint — upload a reference audio file to clone a voice."""
        return await handle_voice_clone(
            input_text=input,
            ref_audio_file=ref_audio,
            ref_text=ref_text,
            response_format=response_format,
            language=language,
            synthesizer=app.state.synthesizer,
        )

    return app
