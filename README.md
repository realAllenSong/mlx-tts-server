# mlx-tts-server

OpenAI-compatible Text-to-Speech server for Apple Silicon, powered by
**Qwen3-TTS** and [mlx-audio](https://github.com/Blaizzy/mlx-audio).

Runs natively on Metal — no CUDA, no cloud, just your Mac.

[中文文档](./docs/README_zh.md)

## Installation

We recommend [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management:

```bash
uv pip install mlx-tts-server
```

Or with pip:

```bash
pip install mlx-tts-server
```

To install from source for development:

```bash
git clone https://github.com/realAllenSong/mlx-tts-server.git
cd mlx-tts-server
uv venv && source .venv/bin/activate
uv pip install -e .
```

For MP3, OPUS, or AAC output, install ffmpeg:

```bash
brew install ffmpeg
```

## Quick Start

Start the TTS server:

```bash
mlx-tts serve mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit --port 8000
```

Synthesize speech:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello, Apple Silicon!","voice":"ryan","response_format":"wav"}' \
  --output speech.wav

open speech.wav
```

## Python Client

Use the [OpenAI Python client](https://github.com/openai/openai-python):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.audio.speech.create(
    model="tts-1",
    input="Hello from Apple Silicon!",
    voice="alloy",
    response_format="wav",
)
response.stream_to_file("output.wav")
```

Or use `httpx` / `requests` directly:

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/v1/audio/speech",
    json={"model": "tts-1", "input": "Hello!", "voice": "ryan", "response_format": "wav"},
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## Voice Cloning

Clone a voice from a short reference audio clip (requires a **Base** model):

```bash
mlx-tts serve mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --port 8000

curl -X POST http://localhost:8000/v1/audio/clone \
  -F "input=Say this in my voice." \
  -F "ref_audio=@/path/to/sample.wav" \
  -F "ref_text=This is a sample." \
  --output cloned.wav
```

## Voice Design

Create a voice from a text description (requires a **VoiceDesign** model):

```bash
mlx-tts serve mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 --port 8000

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Welcome to our app.","voice":"custom","instruct":"A warm, friendly female voice with a slight British accent"}' \
  --output designed.wav
```

## Supported Models

| Model ID | Size | Type |
|---|---|---|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | 0.6B 4-bit | 9 preset voices |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` | 1.7B 4-bit | 9 preset voices |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | 0.6B bf16 | 9 preset voices |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | 1.7B bf16 | 9 preset voices |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` | 0.6B 4-bit | Voice cloning |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | 1.7B bf16 | Voice cloning |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | 1.7B bf16 | Voice design |

## Available Voices (CustomVoice models)

| Voice Name | Gender | OpenAI Alias |
|---|---|---|
| `ryan` | Male | `echo` |
| `aiden` | Male | `nova` |
| `eric` | Male | `onyx` |
| `dylan` | Male | — |
| `serena` | Female | `alloy` |
| `vivian` | Female | `fable` |
| `sohee` | Female | `shimmer` |
| `uncle_fu` | Special | — |
| `ono_anna` | Special | — |

OpenAI voice aliases (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) are
accepted and automatically mapped to the corresponding Qwen3-TTS voice.

## API Reference

### `POST /v1/audio/speech`

Synthesize speech from text.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | required | Model ID (any value, server uses loaded model) |
| `input` | string | required | Text to synthesize |
| `voice` | string | `"ryan"` | Voice name or OpenAI alias |
| `response_format` | string | `"wav"` | `wav`, `mp3`, `flac`, `opus`, `aac`, `pcm` |
| `speed` | float | `1.0` | Playback speed (0.25–4.0) |
| `language` | string | auto | Language (auto-detected if omitted) |
| `instruct` | string | — | Emotion/style instruction or voice description |

### `POST /v1/audio/clone`

Voice cloning via reference audio upload (multipart form).

| Field | Type | Description |
|---|---|---|
| `input` | string | Text to synthesize |
| `ref_audio` | file | Reference audio file (~3s WAV/MP3) |
| `ref_text` | string | Transcript of reference audio |
| `response_format` | string | Output format (default: `wav`) |
| `language` | string | Language (default: `english`) |

### `GET /v1/models`

OpenAI-compatible model listing.

### `GET /v1/audio/speech/voices`

List available voice names for the loaded model.

### `GET /health`

Health check. Returns `{"status": "ok", "model": "...", "uptime": 42.0}`.

## CLI Reference

```
mlx-tts serve MODEL [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host to bind |
| `--port` | `8000` | Port to listen on |
| `--workers` | `1` | Number of uvicorn workers |
| `--log-level` | `info` | Log level (`debug`, `info`, `warning`, `error`) |

## Output Formats

| Format | MIME Type | Requires |
|---|---|---|
| `wav` | `audio/wav` | built-in (soundfile) |
| `flac` | `audio/flac` | built-in (soundfile) |
| `pcm` | `audio/pcm` | built-in |
| `mp3` | `audio/mpeg` | `brew install ffmpeg` |
| `opus` | `audio/opus` | `brew install ffmpeg` |
| `aac` | `audio/aac` | `brew install ffmpeg` |

## License

Apache-2.0
