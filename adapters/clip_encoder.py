"""Local CLIP, loaded in-process. Satisfies the ClipEncoder port."""

from __future__ import annotations

import clip
import numpy as np
import torch
from PIL import Image


class ClipEncoder:
    """OpenAI CLIP, the same weights home2's ImageEmbedder loads."""

    def __init__(self, model_name: str = "ViT-B/32", device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    @property
    def dim(self) -> int:
        return self.model.text_projection.shape[1]

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a string, L2-normalized."""
        with torch.no_grad():
            vector = self.model.encode_text(clip.tokenize([text]).to(self.device))[0].float()
        return (vector / vector.norm()).cpu().numpy()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode an image, L2-normalized. Local only: on the robot vision embeds the crops."""
        with torch.no_grad():
            pixels = self.preprocess(image).unsqueeze(0).to(self.device)
            vector = self.model.encode_image(pixels)[0].float()
        return (vector / vector.norm()).cpu().numpy()
