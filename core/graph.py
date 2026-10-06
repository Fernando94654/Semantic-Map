"""SemanticGraph: all the logic, with no ROS and no I/O.

Two ways in: `from_snapshot` loads a graph that was already built, `load_layout`
plus `observe` builds it online. Not thread-safe on purpose — the ROS adapter
serializes access with a single mutually-exclusive callback group.
"""

from __future__ import annotations

import math
from dataclasses import astuple
from typing import Any, Optional

from core.config import SemanticMapConfig
from core.serialize import rooms_from_layout
from core.types import (
    Discrepancy,
    Observation,
    ObjectInstance,
    ObjectMatch,
    ParsedQuery,
    PointLocation,
    Room,
    Snapshot,
    StaleSurface,
    Surface,
    SurfaceType,
)
from ports import LLM, ClipEncoder, Clock

PARSE_PROMPT = """Split a household robot command into the object it asks for and where to look.
Every value must be in English: translate the command first if it is in another language.
- target: the object wanted, with its attributes. Empty if it is only described by its location.
- anchor: the object or piece of furniture it is located relative to. Never a room. Empty if none.
- relation: "on" or "next_to" when there is an anchor, otherwise empty.
- room: one of {rooms} when the command names a room, otherwise empty.

"bring me the red drink in the kitchen" -> target "red drink", anchor "", relation "", room "kitchen"
"lo que está al lado de la cama" -> target "", anchor "bed", relation "next_to", room ""
"dame la taza que está sobre la mesa de la sala" -> target "cup", anchor "table", relation "on", room "living_room"

Command: {text}"""

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "anchor": {"type": "string"},
        "relation": {"type": "string", "enum": ["", "on", "next_to"]},
        "room": {"type": "string"},
    },
    "required": ["target", "anchor", "relation", "room"],
}


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
        self._text_vectors: dict[str, Any] = {}

    # --- construction ---

    @classmethod
    def from_snapshot(cls, snapshot: Snapshot, **adapters: Any) -> "SemanticGraph":
        """Load a captured graph, embedding whatever the snapshot left empty."""
        graph = cls(**adapters)
        if graph.encoder and snapshot.embedding_dim not in (0, graph.encoder.dim):
            raise ValueError(
                f"snapshot embeddings are {snapshot.embedding_dim}-d, "
                f"the encoder is {graph.encoder.dim}-d"
            )
        graph.rooms, graph.objects = snapshot.rooms, snapshot.objects
        graph.map_name, graph.t_capture = snapshot.map_name, snapshot.t_capture
        graph._embed_missing()
        return graph

    def load_layout(self, areas: dict[str, Any], meta: Optional[dict[str, Any]] = None) -> None:
        """Load the fixed layer from a raw areas.json dict."""
        self.rooms = rooms_from_layout(areas, self.config, meta)

    # --- online input ---

    def observe(self, obs: Observation) -> Optional[str]:
        """Fuse one detection. Returns the instance id, or None if gated out."""
        if obs.confidence < self.config.confidence_min:
            return None
        obj = next((o for o in self.objects.values() if self._same(o, obs)), None)
        if obj is None:
            obj = ObjectInstance(
                id=f"obj_{len(self.objects) + 1:03d}",
                label=obs.label,
                position=obs.position,
                first_seen=obs.stamp,
                observation_count=0,
            )
            self.objects[obj.id] = obj
        where = self.area_for_point(*astuple(obs.position))
        obj.position, obj.area, obj.sublocation = obs.position, where.area, where.sublocation
        obj.category = obs.category or obj.category
        obj.confidence, obj.last_seen = obs.confidence, obs.stamp
        obj.embedding = obs.embedding if obs.embedding is not None else self._embed(obs.label)
        obj.observation_count += 1
        return obj.id

    def mark_scanned(self, area: str, sublocation: str) -> None:
        """Record that the robot just looked at this surface from close enough."""
        self.rooms[area].surfaces[sublocation].last_scanned = self._now()

    # --- geometry ---

    def area_for_point(self, x: float, y: float, z: float = 0.0) -> PointLocation:
        """Resolve a point to a room and its nearest surface.

        Returns an empty area when the point falls outside every polygon:
        admitting it is unknown beats naming the wrong room.
        """
        room = next((r for r in self.rooms.values() if _inside(r.polygon, x, y)), None)
        if room is None:
            return PointLocation()
        where = PointLocation(area=room.name, in_house=True)
        # Something lying on the floor belongs to a floor surface, never to furniture.
        on_floor = z < self.config.height_by_type[SurfaceType.FLOOR.value]
        for surface in room.surfaces.values():
            distance = math.hypot(surface.pose.x - x, surface.pose.y - y)
            if (
                distance < min(where.distance_m, self.config.max_sublocation_distance_m)
                and (surface.type is SurfaceType.FLOOR) == on_floor
            ):
                where.sublocation, where.distance_m = surface.name, distance
        return where

    # --- queries ---

    def objects_at(self, area: str, sublocation: str = "") -> list[ObjectInstance]:
        return [
            o
            for o in self.objects.values()
            if o.area == area and sublocation in ("", o.sublocation)
        ]

    def misplaced(self) -> list[Discrepancy]:
        """Objects whose category does not match the surface they were found on."""
        found = []
        for o in self.objects.values():
            expected = self.config.category_to_furniture.get(o.category)
            if expected and [o.area, o.sublocation] != expected:
                found.append(Discrepancy(o.id, o.label, o.area, o.sublocation, *expected))
        return found

    def stale(self, older_than_s: Optional[float] = None) -> list[StaleSurface]:
        """Surfaces not looked at recently, oldest first; never seen comes first."""
        limit = self.config.stale_after_s if older_than_s is None else older_than_s
        now = self._now()
        ages = [
            StaleSurface(
                s.area, s.name, s.last_scanned,
                math.inf if s.last_scanned is None else now - s.last_scanned,
            )
            for s in self._surfaces()
        ]
        return sorted((s for s in ages if s.age_s > limit), key=lambda s: -s.age_s)

    def find(self, query: str, k: Optional[int] = None) -> list[ObjectMatch]:
        """Retrieve objects by free text; without an encoder, by exact label or category."""
        ranked = self._rank(query, list(self.objects.values()), _appearance)
        return [ObjectMatch(o, score) for o, score in ranked[: k or self.config.top_k]]

    # --- commands: parse is the slow LLM call, select is pure ---

    def parse(self, text: str) -> ParsedQuery:
        """Split a command into target, anchor, relation and room."""
        if self.llm is None:
            return ParsedQuery(target=text)
        rooms = [r.name for r in self.rooms.values() if r.polygon]
        answer = self.llm.complete_json(PARSE_PROMPT.format(rooms=rooms, text=text), PARSE_SCHEMA)
        parsed = ParsedQuery(**{k: answer.get(k, "") for k in PARSE_SCHEMA["properties"]})
        if parsed.room not in rooms:
            parsed.room = ""
        return parsed

    def select(self, parsed: ParsedQuery) -> list[ObjectMatch]:
        """Resolve a parsed command to objects.

        Room and relation are settled by geometry. The anchor is matched by
        name, text against text; the target by appearance, text against the
        object embedding.
        """
        objects = [o for o in self.objects.values() if parsed.room in ("", o.area)]
        if parsed.anchor:
            surfaces = [s for s in self._surfaces() if parsed.room in ("", s.area)]
            anchors = self._rank(parsed.anchor, surfaces + objects, self._name_vector)
            if not anchors:
                return []
            anchor = anchors[0][0]
            objects = [
                o for o in objects if o is not anchor and self._related(o, anchor, parsed.relation)
            ]
        if not parsed.target:
            return [ObjectMatch(o, 1.0) for o in objects]
        ranked = self._rank(parsed.target, objects, _appearance)
        return [ObjectMatch(o, score) for o, score in ranked[: self.config.top_k]]

    def resolve(self, text: str) -> list[ObjectMatch]:
        return self.select(self.parse(text))

    # --- internals ---

    def _now(self) -> float:
        return self.clock.now() if self.clock is not None else self.t_capture

    def _surfaces(self) -> list[Surface]:
        return [s for room in self.rooms.values() for s in room.surfaces.values()]

    def _embed(self, name: str):
        """CLIP text vector of a name or query, computed once."""
        if self.encoder is None:
            return None
        if name not in self._text_vectors:
            caption = self.config.query_template.format(name.replace("_", " "))
            self._text_vectors[name] = self.encoder.encode_text(caption)
        return self._text_vectors[name]

    def _embed_missing(self) -> None:
        """Objects that arrive without an embedding get one from their label."""
        for obj in self.objects.values():
            if obj.embedding is None:
                obj.embedding = self._embed(obj.label)

    def _name_vector(self, node: Any):
        return self._embed(_name(node))

    def _rank(self, text: str, nodes: list, vector: Any) -> list[tuple[Any, float]]:
        """Score nodes against a text, best first; `vector` picks what each node is compared by."""
        if self.encoder is None:
            wanted = text.lower().replace(" ", "_")
            return [
                (n, 1.0) for n in nodes if wanted in (_name(n), getattr(n, "category", None))
            ]
        query = self._embed(text)
        scored = [(n, float(vector(n) @ query)) for n in nodes if vector(n) is not None]
        scored = [pair for pair in scored if pair[1] >= self.config.sim_min]
        return sorted(scored, key=lambda pair: -pair[1])

    def _related(self, obj: ObjectInstance, anchor: Any, relation: str) -> bool:
        """True when the object is on the anchor surface or, unless "on", near the anchor."""
        if isinstance(anchor, Surface):
            if (obj.area, obj.sublocation) == (anchor.area, anchor.name):
                return True
            if relation == "on":
                return False
            x, y = anchor.pose.x, anchor.pose.y
        else:
            x, y = anchor.position.x, anchor.position.y
        return math.hypot(obj.position.x - x, obj.position.y - y) <= self.config.near_m

    def _same(self, obj: ObjectInstance, obs: Observation) -> bool:
        """True when a detection is another view of an instance already in the graph."""
        if math.dist(astuple(obj.position), astuple(obs.position)) > self.config.merge_dist_m:
            return False
        if obj.embedding is None or obs.embedding is None:
            return obj.label == obs.label
        return float(obj.embedding @ obs.embedding) >= self.config.merge_sim_min


def _appearance(obj: ObjectInstance):
    return obj.embedding


def _name(node: Any) -> str:
    """Objects have a label, surfaces a name."""
    return node.label if isinstance(node, ObjectInstance) else node.name


def _inside(polygon: list[tuple[float, float]], x: float, y: float) -> bool:
    """Ray casting point-in-polygon."""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
