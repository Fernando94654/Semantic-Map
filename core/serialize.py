"""Parsing of the on-disk dicts into typed rooms and objects. No I/O."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from core.config import SemanticMapConfig
from core.types import ObjectInstance, Point, Pose, Room, Snapshot, Surface, SurfaceType

NOT_SURFACES = ("polygon", "safe_place")


def rooms_from_layout(
    layout: dict[str, Any],
    config: SemanticMapConfig,
    meta: Optional[dict[str, Any]] = None,
    scanned: Optional[dict[str, float]] = None,
) -> dict[str, Room]:
    """Build rooms and surfaces from an areas.json dict."""
    meta, scanned = meta or {}, scanned or {}
    rooms = {}
    for area, entry in layout.items():
        room = Room(area, [tuple(p) for p in entry.get("polygon", [])])
        if "safe_place" in entry:
            room.safe_place = Pose(*entry["safe_place"])
        for name, pose in entry.items():
            if name in NOT_SURFACES:
                continue
            info = meta.get(f"{area}/{name}", {})
            kind = SurfaceType(info.get("type") or _type_from_name(name, config))
            room.surfaces[name] = Surface(
                name=name,
                area=area,
                pose=Pose(*pose),
                type=kind,
                height_m=info.get("height_m", config.height_by_type[kind.value]),
                front_yaw_rad=info.get("front_yaw_rad"),
                arm_pose=info.get("arm_pose"),
                last_scanned=scanned.get(f"{area}/{name}"),
            )
        rooms[area] = room
    return rooms


def snapshot_from_dict(data: dict[str, Any], config: SemanticMapConfig) -> Snapshot:
    """Parse a snapshot dict; object keys match the ObjectInstance fields."""
    objects = {}
    for raw in data.get("objects", []):
        embedding = raw["embedding"]
        objects[raw["id"]] = ObjectInstance(
            **{
                **raw,
                "position": Point(*raw["position"]),
                "embedding": None if embedding is None else np.asarray(embedding, np.float32),
            }
        )
    return Snapshot(
        rooms=rooms_from_layout(
            data["layout"], config, data.get("furniture_meta"), data.get("scanned")
        ),
        objects=objects,
        t_capture=data["t_capture"],
        map_name=data.get("map_name", ""),
        embedding_dim=data.get("embedding_dim", 0),
    )


def _type_from_name(name: str, config: SemanticMapConfig) -> str:
    """Guess the surface type from its name; the longest matching fragment wins."""
    fragments = [f for f in config.type_by_name if f in name]
    return config.type_by_name[max(fragments, key=len)] if fragments else config.default_type
