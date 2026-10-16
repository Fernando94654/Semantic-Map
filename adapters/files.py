"""File loading. The only place that opens files; the core never does."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.config import SemanticMapConfig
from core.serialize import snapshot_from_dict
from core.types import Snapshot

EMBEDDINGS_NAME = "embeddings.npz"


def load_snapshot(path: str | Path, config: SemanticMapConfig) -> Snapshot:
    """Read a snapshot, with the image embeddings beside it when they have been built."""
    path = Path(path)
    with open(path) as file:
        snapshot = snapshot_from_dict(json.load(file), config)
    embeddings = path.parent / EMBEDDINGS_NAME
    if embeddings.exists():
        with np.load(embeddings) as vectors:
            for object_id in vectors.files:
                if object_id in snapshot.objects:
                    snapshot.objects[object_id].embedding = vectors[object_id]
    return snapshot
