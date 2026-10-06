"""File loading. The only place that opens files; the core never does."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import SemanticMapConfig
from core.serialize import snapshot_from_dict
from core.types import Snapshot


def load_snapshot(path: str | Path, config: SemanticMapConfig) -> Snapshot:
    with open(path) as file:
        return snapshot_from_dict(json.load(file), config)
