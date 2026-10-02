"""SemanticGraph: all the logic, with no ROS and no I/O.

Two ways in: `from_snapshot` loads a graph that was already built, `load_layout`
plus `observe` builds it online. Not thread-safe on purpose — the ROS adapter
serializes access with a single mutually-exclusive callback group.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from core.config import SemanticMapConfig
from core.types import (
    Discrepancy,
    Observation,
    ObjectInstance,
    ObjectMatch,
    PointLocation,
    Room,
    Snapshot,
    StaleSurface,
    Surface,
)
from ports import Clock, ClipEncoder, LLM


class SemanticGraph:
    """A three-layer map of the house: rooms, furniture, and the objects on them."""

    def __init__(
        self,
        config: Optional[SemanticMapConfig] = None,
        encoder: Optional[ClipEncoder] = None,
        llm: Optional[LLM] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.config = config or SemanticMapConfig()
        self.encoder = encoder
        self.llm = llm
        self.clock = clock

        self.rooms: dict[str, Room] = {}
        self.objects: dict[str, ObjectInstance] = {}
        self.map_name: str = ""
        self.t_capture: float = 0.0

    # --- construction ---

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot, **kwargs: Any) -> "SemanticGraph":
        raise NotImplementedError

    def load_layout(self, areas: dict[str, Any], meta: Optional[dict[str, Any]] = None) -> None:
        """Load the fixed layer from a raw areas.json dict.

        Areas without a polygon are referee waypoints. In each area, every key
        other than `polygon` and `safe_place` is furniture.
        """
        raise NotImplementedError

    def to_snapshot(self) -> Snapshot:
        raise NotImplementedError

    def reset_objects(self) -> None:
        """Drop the object layer, keep the layout, after a localization reset."""
        raise NotImplementedError

    # --- online input ---

    def observe(self, obs: Observation) -> Optional[str]:
        """Fuse one detection. Returns the instance id, or None if gated out."""
        raise NotImplementedError

    def observe_many(self, observations: Iterable[Observation]) -> list[str]:
        raise NotImplementedError

    def mark_scanned(self, area: str, sublocation: str, stamp: Optional[float] = None) -> None:
        """Record that the robot looked at this surface from close enough."""
        raise NotImplementedError

    # --- geometry ---

    def area_for_point(self, x: float, y: float, z: float = 0.0) -> PointLocation:
        """Resolve a point to a room and its nearest furniture.

        Returns an empty area when the point falls outside every polygon:
        admitting it is unknown beats naming the wrong room.
        """
        raise NotImplementedError

    def surfaces(self, area: str = "", sublocation: str = "") -> list[Surface]:
        raise NotImplementedError

    # --- semantic queries ---

    def find(self, query: str, k: Optional[int] = None) -> list[ObjectMatch]:
        """Retrieve objects by free text, via the encoder when there is one."""
        raise NotImplementedError

    def objects_at(self, area: str, sublocation: str = "") -> list[ObjectInstance]:
        raise NotImplementedError

    def misplaced(self) -> list[Discrepancy]:
        """Objects whose category does not match the surface they were found on."""
        raise NotImplementedError

    def stale(self, older_than_s: Optional[float] = None) -> list[StaleSurface]:
        """Surfaces not looked at recently, oldest first."""
        raise NotImplementedError

    # --- LLM boundary, split so the slow call happens outside the lock ---

    def to_prompt(self, focus: Optional[str] = None, max_objects: int = 40) -> str:
        """Serialize the graph as compact text for the LLM. Fast, no I/O."""
        raise NotImplementedError

    def apply_plan(self, plan: dict[str, Any]) -> Any:
        """Validate an LLM plan against the graph and resolve it to real nodes."""
        raise NotImplementedError

    # --- internals ---

    def _now(self) -> float:
        return self.clock.now() if self.clock is not None else self.t_capture
