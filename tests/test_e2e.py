# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests: real model + live server + HTTP calls."""

from __future__ import annotations

import threading
import time

import httpx
import numpy as np
import pytest
import uvicorn

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
SERVER_PORT = 18765
SERVER_BASE = f"http://127.0.0.1:{SERVER_PORT}"


@pytest.fixture(scope="module")
def tts_server():
    """Start a real TTS server with a real model in a background thread."""
    from mlx_tts_server.loader import load_tts_model
    from mlx_tts_server.server.app import create_app

    synthesizer = load_tts_model(MODEL_ID)
    app = create_app(model_id=MODEL_ID, synthesizer=synthesizer)

    config = uvicorn.Config(app, host="127.0.0.1", port=SERVER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(30):
        try:
            r = httpx.get(f"{SERVER_BASE}/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        pytest.fail("TTS server failed to start within 30 seconds")

    yield SERVER_BASE

    server.should_exit = True
    thread.join(timeout=5)


@pytest.mark.slow
def test_e2e_health(tts_server):
    resp = httpx.get(f"{tts_server}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == MODEL_ID


@pytest.mark.slow
def test_e2e_list_models(tts_server):
    resp = httpx.get(f"{tts_server}/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL_ID


@pytest.mark.slow
def test_e2e_list_voices(tts_server):
    resp = httpx.get(f"{tts_server}/v1/audio/speech/voices")
    assert resp.status_code == 200
    voices = resp.json()["voices"]
    assert len(voices) > 0
    assert "ryan" in voices


@pytest.mark.slow
def test_e2e_synthesize_wav(tts_server):
    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "Hello from Apple Silicon! This is a test of Qwen3-TTS.",
            "voice": "ryan",
            "response_format": "wav",
        },
        timeout=180,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content[:4] == b"RIFF"
    duration = float(resp.headers["x-audio-duration"])
    assert duration > 0.5, f"Expected > 0.5s audio, got {duration:.2f}s"


@pytest.mark.slow
def test_e2e_synthesize_flac(tts_server):
    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "FLAC format test.",
            "voice": "serena",
            "response_format": "flac",
        },
        timeout=180,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/flac")
    assert resp.content[:4] == b"fLaC"


@pytest.mark.slow
def test_e2e_openai_voice_alias(tts_server):
    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "Testing OpenAI voice alias.",
            "voice": "echo",
            "response_format": "wav",
        },
        timeout=180,
    )
    assert resp.status_code == 200
    assert resp.content[:4] == b"RIFF"


@pytest.mark.slow
def test_e2e_pcm_format(tts_server):
    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "PCM test.",
            "voice": "aiden",
            "response_format": "pcm",
        },
        timeout=180,
    )
    assert resp.status_code == 200
    assert len(resp.content) % 4 == 0
    audio = np.frombuffer(resp.content, dtype=np.float32)
    assert len(audio) > 0
    assert np.all(np.isfinite(audio))


@pytest.mark.slow
def test_e2e_empty_input_rejected(tts_server):
    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "   ",
            "voice": "ryan",
            "response_format": "wav",
        },
        timeout=30,
    )
    assert resp.status_code in (400, 422)


@pytest.mark.slow
def test_e2e_mp3_if_ffmpeg_available(tts_server):
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")

    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "MP3 format test.",
            "voice": "ryan",
            "response_format": "mp3",
        },
        timeout=180,
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert len(resp.content) > 100


@pytest.mark.slow
def test_e2e_synthesized_audio_is_valid_waveform(tts_server):
    resp = httpx.post(
        f"{tts_server}/v1/audio/speech",
        json={
            "model": MODEL_ID,
            "input": "The quick brown fox jumps over the lazy dog.",
            "voice": "ryan",
            "response_format": "pcm",
        },
        timeout=180,
    )
    assert resp.status_code == 200
    audio = np.frombuffer(resp.content, dtype=np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    assert rms > 0.001, f"Audio looks silent: RMS={rms:.6f}"
