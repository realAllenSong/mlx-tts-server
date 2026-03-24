# SPDX-License-Identifier: Apache-2.0
"""Request/response types for Text-to-Speech."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, field_validator

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class SpeechSynthesisRequest(BaseModel):
    """OpenAI-compatible /v1/audio/speech request."""
    model: str
    input: str
    voice: str = "ryan"
    response_format: ResponseFormat = "wav"
    speed: float = 1.0
    language: str | None = None
    instruct: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None

    @field_validator("model")
    @classmethod
    def model_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be empty")
        return v

    @field_validator("input")
    @classmethod
    def input_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input text must not be empty")
        return v

    @field_validator("speed")
    @classmethod
    def valid_speed(cls, v: float) -> float:
        if not (0.25 <= v <= 4.0):
            raise ValueError("speed must be between 0.25 and 4.0")
        return v


@dataclass
class SpeechSynthesisResult:
    """Result of a speech synthesis operation."""
    audio: np.ndarray
    sample_rate: int
    duration: float
    voice: str = ""
    model: str = ""
