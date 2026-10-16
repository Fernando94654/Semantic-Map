"""Embed data/images with CLIP into data/embeddings.npz, one vector per object id.

    python scripts/embed_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters.clip_encoder import ClipEncoder  # noqa: E402
from adapters.files import EMBEDDINGS_NAME  # noqa: E402

IMAGES = ROOT / "data" / "images"


def embed_images(encoder: ClipEncoder) -> dict[str, np.ndarray]:
    """Image embedding of every photo, keyed by the object id in its file name."""
    return {p.stem: encoder.encode_image(Image.open(p)) for p in sorted(IMAGES.glob("*.jpg"))}


if __name__ == "__main__":
    vectors = embed_images(ClipEncoder())
    np.savez(ROOT / "data" / EMBEDDINGS_NAME, **vectors)
    print(f"{len(vectors)} embeddings -> data/{EMBEDDINGS_NAME}")
