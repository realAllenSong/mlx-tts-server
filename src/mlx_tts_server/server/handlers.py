# SPDX-License-Identifier: Apache-2.0
"""Request handlers for the TTS API server."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import Response

from mlx_tts_server.audio_output import MIME_TYPES, encode_audio
from mlx_tts_server.protocol import SpeechSynthesisRequest

logger = logging.getLogger(__name__)


async def handle_voice_clone(
    input_text: str,
    ref_audio_file: UploadFile,
    ref_text: str,
    response_format: str,
    language: str,
    synthesizer: object,
) -> Response:
    """Handle POST /v1/audio/clone — voice cloning via reference audio upload."""
    suffix = Path(ref_audio_file.filename or "ref.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        content = await ref_audio_file.read()
        tmp.write(content)

    t0 = time.monotonic()
    try:
        audio, sample_rate = synthesizer.generate(
            input_text,
            ref_audio=tmp_path,
            ref_text=ref_text,
            language=language,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Voice cloning failed")
        raise HTTPException(status_code=500, detail=f"Voice cloning error: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    elapsed = time.monotonic() - t0
    duration = len(audio) / sample_rate

    logger.info(
        "TTS clone: synthesized %.2fs audio in %.2fs (RTF=%.2f) fmt=%r",
        duration, elapsed, elapsed / max(duration, 1e-6), response_format,
    )

    try:
        audio_bytes = encode_audio(audio, sample_rate, response_format)
    except Exception as e:
        logger.exception("Audio encoding failed")
        raise HTTPException(status_code=500, detail=f"Encoding error: {e}") from e

    content_type = MIME_TYPES.get(response_format, "application/octet-stream")
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Generation-Time": f"{elapsed:.3f}",
        },
    )


async def handle_speech_synthesis(
    request: SpeechSynthesisRequest,
    synthesizer: object,
) -> Response:
    """Handle POST /v1/audio/speech."""
    t0 = time.monotonic()
    try:
        audio, sample_rate = synthesizer.generate(
            request.input,
            voice=request.voice,
            language=request.language,
            instruct=request.instruct,
            speed=request.speed,
            ref_audio=request.ref_audio,
            ref_text=request.ref_text,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {e}") from e

    elapsed = time.monotonic() - t0
    duration = len(audio) / sample_rate

    logger.info(
        "TTS: synthesized %.2fs audio in %.2fs (RTF=%.2f) voice=%r fmt=%r",
        duration, elapsed, elapsed / max(duration, 1e-6), request.voice, request.response_format,
    )

    fmt = request.response_format
    try:
        audio_bytes = encode_audio(audio, sample_rate, fmt)
    except Exception as e:
        logger.exception("Audio encoding failed")
        raise HTTPException(status_code=500, detail=f"Encoding error: {e}") from e

    content_type = MIME_TYPES.get(fmt, "application/octet-stream")
    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Generation-Time": f"{elapsed:.3f}",
        },
    )
