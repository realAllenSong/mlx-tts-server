# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for the `mlx-tts` command."""

from __future__ import annotations

import logging
import sys

import click
import uvicorn


@click.group()
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    show_default=True,
    help="Logging level.",
)
def cli(log_level: str) -> None:
    """mlx-tts: OpenAI-compatible Text-to-Speech server for Apple Silicon.

    Powered by mlx-audio and Qwen3-TTS.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@cli.command("serve")
@click.argument("model")
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind.")
@click.option("--port", default=8000, show_default=True, help="Port to listen on.")
@click.option(
    "--workers",
    default=1,
    show_default=True,
    help="Number of uvicorn worker processes.",
)
def serve(model: str, host: str, port: int, workers: int) -> None:
    """Start a TTS server for MODEL.

    MODEL is a HuggingFace model ID or local path.

    \b
    Examples:
      mlx-tts serve mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit
      mlx-tts serve mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 --port 8001
      mlx-tts serve ~/models/Qwen3-TTS-local
    """
    from mlx_tts_server.loader import load_tts_model
    from mlx_tts_server.server.app import create_app

    click.echo(f"Loading TTS model: {model}")
    try:
        synthesizer = load_tts_model(model)
    except Exception as e:
        click.echo(f"ERROR: Failed to load model {model!r}: {e}", err=True)
        sys.exit(1)

    click.echo(f"\nTTS server running at http://{host}:{port}")
    click.echo("  POST /v1/audio/speech        — synthesize speech (OpenAI-compatible)")
    click.echo("  POST /v1/audio/clone         — voice cloning (multipart upload)")
    click.echo("  GET  /v1/models              — list models")
    click.echo("  GET  /v1/audio/speech/voices  — list voices")
    click.echo("  GET  /health                 — health check")
    click.echo("\nPress Ctrl+C to stop.\n")

    app = create_app(model_id=model, synthesizer=synthesizer)

    if workers > 1:
        click.echo(
            "WARNING: --workers > 1 is not supported with MLX (Metal is not fork-safe). "
            "Forcing workers=1.",
            err=True,
        )
        workers = 1

    uvicorn.run(app, host=host, port=port, workers=workers, log_level="warning")


def main() -> None:
    """Entry point for `mlx-tts` command."""
    cli()


if __name__ == "__main__":
    main()
