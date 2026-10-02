"""Local CLIP, loaded in-process. Satisfies the ClipEncoder port."""

from __future__ import annotations

import numpy as np


class ClipEncoder:
    """CLIP running on this machine, image and text in one space."""

    def __init__(self, model_name: str = "ViT-B-32", device: str = "cpu") -> None:
        pass

    @property
    def dim(self) -> int:
        """Embedding size, 512 for ViT-B/32."""
        pass

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a query string, L2-normalized."""
        pass

    def encode_image(self, crop: np.ndarray) -> np.ndarray:
        """Encode an image crop, L2-normalized."""
        pass
