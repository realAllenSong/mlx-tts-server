# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TTS FastAPI server (no real model required)."""

import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"


def _make_mock_synthesizer(
    voices=None,
    audio=None,
    sample_rate=24000,
):
    synth = MagicMock()
    synth.voices = voices or ["ryan", "serena", "aiden"]
    synth.config.sample_rate = sample_rate
    if audio is None:
        audio = np.zeros(24000, dtype=np.float32)
    synth.generate.return_value = (audio, sample_rate)
    return synth


@pytest.fixture
def client():
    from mlx_tts_server.server.app import create_app
    synth = _make_mock_synthesizer()
    app = create_app(model_id=MODEL_ID, synthesizer=synth)
    return TestClient(app, raise_server_exceptions=True)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == MODEL_ID


def test_models_endpoint(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL_ID
    assert data["data"][0]["capabilities"]["audio_generation"] is True


def test_voices_endpoint(client):
    resp = client.get("/v1/audio/speech/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert "voices" in data
    assert "ryan" in data["voices"]


def test_speech_wav(client):
    resp = client.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "input": "Hello world.",
        "voice": "ryan",
        "response_format": "wav",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content[:4] == b"RIFF"


def test_speech_pcm(client):
    resp = client.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "input": "Hello.",
        "voice": "ryan",
        "response_format": "pcm",
    })
    assert resp.status_code == 200
    assert "pcm" in resp.headers["content-type"]


def test_speech_flac(client):
    resp = client.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "input": "Hello.",
        "voice": "ryan",
        "response_format": "flac",
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/flac")
    assert resp.content[:4] == b"fLaC"


def test_speech_missing_input(client):
    resp = client.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "voice": "ryan",
    })
    assert resp.status_code == 422


def test_speech_invalid_format(client):
    resp = client.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "input": "Hello.",
        "voice": "ryan",
        "response_format": "xyz",
    })
    assert resp.status_code == 422


def test_speech_openai_voice_alias(client):
    synth = _make_mock_synthesizer()
    from mlx_tts_server.server.app import create_app
    app = create_app(model_id=MODEL_ID, synthesizer=synth)
    tc = TestClient(app)
    resp = tc.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "input": "Hello.",
        "voice": "alloy",
        "response_format": "wav",
    })
    assert resp.status_code == 200
    call_kwargs = synth.generate.call_args
    assert call_kwargs is not None


def test_speech_x_audio_duration_header(client):
    resp = client.post("/v1/audio/speech", json={
        "model": MODEL_ID,
        "input": "Hello.",
        "voice": "ryan",
        "response_format": "wav",
    })
    assert resp.status_code == 200
    assert "x-audio-duration" in resp.headers
    duration = float(resp.headers["x-audio-duration"])
    assert duration > 0


def test_speech_synthesis_error_returns_500(client):
    from mlx_tts_server.server.app import create_app
    synth = _make_mock_synthesizer()
    synth.generate.side_effect = RuntimeError("GPU out of memory")
    app = create_app(model_id=MODEL_ID, synthesizer=synth)
    tc = TestClient(app, raise_server_exceptions=False)
    resp = tc.post("/v1/audio/speech", json={
        "model": MODEL_ID, "input": "Hello.", "voice": "ryan", "response_format": "wav"
    })
    assert resp.status_code == 500


def test_speech_empty_input_returns_400(client):
    from mlx_tts_server.server.app import create_app
    synth = _make_mock_synthesizer()
    synth.generate.side_effect = ValueError("Input text must not be empty")
    app = create_app(model_id=MODEL_ID, synthesizer=synth)
    tc = TestClient(app, raise_server_exceptions=False)
    resp = tc.post("/v1/audio/speech", json={
        "model": MODEL_ID, "input": "   ", "voice": "ryan", "response_format": "wav"
    })
    assert resp.status_code in (400, 422)
