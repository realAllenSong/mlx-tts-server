# mlx-tts-server

OpenAI-compatible Text-to-Speech server for Apple Silicon, powered by
**Qwen3-TTS** and [mlx-audio](https://github.com/Blaizzy/mlx-audio).

Runs natively on Metal — no CUDA, no cloud, just your Mac.

[中文文档](#中文文档)

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

---

# 中文文档

兼容 OpenAI 接口的文字转语音服务器，专为 Apple Silicon 打造，基于
**Qwen3-TTS** 和 [mlx-audio](https://github.com/Blaizzy/mlx-audio)。

原生运行在 Metal 上 — 无需 CUDA，无需云服务，只需要你的 Mac。

## 安装

推荐使用 [uv](https://docs.astral.sh/uv/) 进行包管理（更快、更可靠）：

```bash
uv pip install mlx-tts-server
```

也可以用 pip：

```bash
pip install mlx-tts-server
```

从源码安装（开发模式）：

```bash
git clone https://github.com/realAllenSong/mlx-tts-server.git
cd mlx-tts-server
uv venv && source .venv/bin/activate
uv pip install -e .
```

如需 MP3、OPUS 或 AAC 输出格式，安装 ffmpeg：

```bash
brew install ffmpeg
```

## 快速开始

启动 TTS 服务器：

```bash
mlx-tts serve mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit --port 8000
```

合成语音：

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"你好，Apple Silicon！","voice":"ryan","response_format":"wav"}' \
  --output speech.wav

open speech.wav
```

## Python 客户端

使用 [OpenAI Python 客户端](https://github.com/openai/openai-python)：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.audio.speech.create(
    model="tts-1",
    input="你好，这是在 Apple Silicon 上运行的语音合成！",
    voice="alloy",
    response_format="wav",
)
response.stream_to_file("output.wav")
```

或直接使用 `httpx` / `requests`：

```python
import httpx

resp = httpx.post(
    "http://localhost:8000/v1/audio/speech",
    json={"model": "tts-1", "input": "你好！", "voice": "ryan", "response_format": "wav"},
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

## 语音克隆

从一段短音频中克隆声音（需要 **Base** 模型）：

```bash
mlx-tts serve mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 --port 8000

curl -X POST http://localhost:8000/v1/audio/clone \
  -F "input=用我的声音说这句话。" \
  -F "ref_audio=@/path/to/sample.wav" \
  -F "ref_text=这是一段示例音频。" \
  --output cloned.wav
```

## 语音设计

通过文字描述创建新声音（需要 **VoiceDesign** 模型）：

```bash
mlx-tts serve mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 --port 8000

curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"欢迎使用我们的应用。","voice":"custom","instruct":"温暖友善的女声，带有轻微的播音腔"}' \
  --output designed.wav
```

## 支持的模型

| 模型 ID | 大小 | 类型 |
|---|---|---|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | 0.6B 4-bit | 9 个预设声音 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` | 1.7B 4-bit | 9 个预设声音 |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | 0.6B bf16 | 9 个预设声音 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | 1.7B bf16 | 9 个预设声音 |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit` | 0.6B 4-bit | 语音克隆 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | 1.7B bf16 | 语音克隆 |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | 1.7B bf16 | 语音设计 |

## 可用声音（CustomVoice 模型）

| 声音名称 | 性别 | OpenAI 别名 |
|---|---|---|
| `ryan` | 男 | `echo` |
| `aiden` | 男 | `nova` |
| `eric` | 男 | `onyx` |
| `dylan` | 男 | — |
| `serena` | 女 | `alloy` |
| `vivian` | 女 | `fable` |
| `sohee` | 女 | `shimmer` |
| `uncle_fu` | 特殊 | — |
| `ono_anna` | 特殊 | — |

兼容 OpenAI 声音别名（`alloy`、`echo`、`fable`、`onyx`、`nova`、`shimmer`），自动映射到对应的 Qwen3-TTS 声音。

## API 参考

### `POST /v1/audio/speech`

文字转语音。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | string | 必填 | 模型 ID（任意值，服务器使用已加载的模型） |
| `input` | string | 必填 | 要合成的文本 |
| `voice` | string | `"ryan"` | 声音名称或 OpenAI 别名 |
| `response_format` | string | `"wav"` | `wav`、`mp3`、`flac`、`opus`、`aac`、`pcm` |
| `speed` | float | `1.0` | 语速（0.25–4.0） |
| `language` | string | 自动检测 | 语言（省略则自动检测） |
| `instruct` | string | — | 情绪/风格指令或声音描述 |

### `POST /v1/audio/clone`

通过上传参考音频进行语音克隆（multipart 表单）。

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 要合成的文本 |
| `ref_audio` | file | 参考音频文件（约 3 秒 WAV/MP3） |
| `ref_text` | string | 参考音频的文字内容 |
| `response_format` | string | 输出格式（默认：`wav`） |
| `language` | string | 语言（默认：`english`） |

### `GET /v1/models`

兼容 OpenAI 的模型列表接口。

### `GET /v1/audio/speech/voices`

列出当前模型可用的声音。

### `GET /health`

健康检查。返回 `{"status": "ok", "model": "...", "uptime": 42.0}`。

## 命令行参考

```
mlx-tts serve MODEL [OPTIONS]
```

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 绑定地址 |
| `--port` | `8000` | 监听端口 |
| `--workers` | `1` | uvicorn 工作进程数 |
| `--log-level` | `info` | 日志级别（`debug`、`info`、`warning`、`error`） |

## 输出格式

| 格式 | MIME 类型 | 依赖 |
|---|---|---|
| `wav` | `audio/wav` | 内置（soundfile） |
| `flac` | `audio/flac` | 内置（soundfile） |
| `pcm` | `audio/pcm` | 内置 |
| `mp3` | `audio/mpeg` | `brew install ffmpeg` |
| `opus` | `audio/opus` | `brew install ffmpeg` |
| `aac` | `audio/aac` | `brew install ffmpeg` |

## 许可证

Apache-2.0
