"""What the core needs from the outside world.

The core depends only on these protocols, never on ROS, torch or the filesystem.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClipEncoder(Protocol):
    """CLIP image and text in one space."""

    @property
    def dim(self) -> int: ...

    def encode_text(self, text: str) -> np.ndarray: ...

    def encode_image(self, crop: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class LLM(Protocol):
    """Structured completion. BAML over Ollama on the robot."""

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class Clock(Protocol):
    """Epoch seconds."""

    def now(self) -> float: ...
