"""Run every experiment and save the raw records to results/<host>.json.

    python -m eval.run                          # full run
    python -m eval.run --repeats 1 --limit 10   # smoke run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from scipy.spatial import cKDTree

from adapters.clip_encoder import ClipEncoder
from adapters.files import load_snapshot
from adapters.llm import OllamaLLM
from core.config import SemanticMapConfig
from core.graph import SemanticGraph
from core.types import ObjectInstance, ParsedQuery, Point, Snapshot
from eval.methods import cascade, clip_only, full_graph, vlm_top_k
from main import MOCK_SNAPSHOT

ROOT = Path(__file__).resolve().parent.parent
QUERIES = json.loads((ROOT / "data" / "queries.json").read_text())
PARSER, VERIFIER = "qwen3:8b", "qwen2.5vl:7b"
# The cascade's operating point: escalation is tuned and reported for this set.
MAIN_ALPHA, MAIN_SET = 0.1, "aps"
# Larger graphs overflow the parser's context in the full-graph baseline.
FULL_GRAPH_MAX = 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="commands per split, 0 for all")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--scales", type=int, nargs="+", default=[60, 500, 5000])
    parser.add_argument("--scale-commands", type=int, default=20)
    args = parser.parse_args()

    config = SemanticMapConfig()
    snapshot = load_snapshot(MOCK_SNAPSHOT, config)
    llm, vlm = OllamaLLM(PARSER), OllamaLLM(VERIFIER)
    graph = SemanticGraph.from_snapshot(
        snapshot, config=config, encoder=ClipEncoder(), llm=llm, vlm=vlm
    )
    calibration = _split(QUERIES, "calibration", args.limit)
    test = _split(QUERIES, "test", args.limit)
    log = Checkpoint(ROOT / "results" / f"{socket.gethostname()}.json", args)

    for repeat in range(args.repeats):
        graph.parse_cache.clear()
        _warm(llm)
        for query in calibration + test:
            done = log.get("parse", None, repeat, query["text"])
            if done:
                graph.parse_cache[query["text"]] = ParsedQuery(**done["parsed"])
            else:
                log.add(_parse(graph, llm, query, repeat))
        for query in test:
            if not log.get("main", "A", repeat, query["text"]):
                log.add(_record("main", "A", graph, query, full_graph(graph, llm, query["text"]), repeat))
            if not log.get("main", "B", repeat, query["text"]):
                log.add(_record("main", "B", graph, query, clip_only(graph, query["text"]), repeat))
        _warm(vlm)
        for query in test:
            if not log.get("main", "C", repeat, query["text"]):
                log.add(_record("main", "C", graph, query, vlm_top_k(graph, vlm, query["text"]), repeat))
        _calibrate(graph, calibration, MAIN_ALPHA, MAIN_SET)
        for query in calibration + test:
            if not log.get("escalation", None, repeat, query["text"]):
                log.add(_escalation(graph, query, repeat))
        tuning = [log.get("escalation", None, repeat, q["text"]) for q in calibration]
        graph.config.escalate_above = _tune(tuning)
        for alpha, method in [(a, m) for a in args.alphas for m in ("threshold", "aps")]:
            cutoff = _calibrate(graph, calibration, alpha, method)
            for query in test:
                if not log.get("main", "D", repeat, query["text"], alpha, method):
                    result = cascade(graph, vlm, query["text"])
                    log.add(_record("main", "D", graph, query, result, repeat,
                                    alpha=alpha, set_method=method, cutoff=cutoff,
                                    escalate_above=graph.config.escalate_above))
        log.save()

    log.deterministic = all(
        _ids(graph.select(graph.parse(q["text"]))) == _ids(graph.select(graph.parse(q["text"])))
        for q in test
    )
    _noise(log, graph, snapshot, calibration, test, args.noise)
    _scaling(log, graph, snapshot, calibration, test[: args.scale_commands], args.scales)
    log.save()
    print(f"{len(log.records)} records -> {log.path.relative_to(ROOT)}")


class Checkpoint:
    """Records saved as they come, so a rerun with the same arguments skips what is done."""

    KEY = ("experiment", "method", "repeat", "text", "alpha", "set_method", "size")
    EVERY = 25

    def __init__(self, path: Path, args) -> None:
        self.path, self.args, self.deterministic = path, args, None
        saved = json.loads(path.read_text()) if path.exists() else {}
        same = saved.get("environment", {}).get("args") == vars(args)
        self.records: list[dict] = saved["records"] if same else []
        self.done = {self._key(r): r for r in self.records}
        print(f"resuming with {len(self.records)} records" if self.records else "starting", flush=True)

    def get(self, experiment, method, repeat, text, alpha=None, set_method=None, size=None):
        return self.done.get((experiment, method, repeat, text, alpha, set_method, size))

    def add(self, record: dict) -> None:
        self.records.append(record)
        self.done[self._key(record)] = record
        if len(self.records) % self.EVERY == 0:
            self.save()

    def save(self) -> None:
        """Write to a temporary file and rename it, so a power cut never leaves half a file."""
        self.path.parent.mkdir(exist_ok=True)
        data = {"environment": _environment(self.args, self.deterministic), "records": self.records}
        partial = self.path.with_suffix(".tmp")
        partial.write_text(json.dumps(data, indent=1))
        os.replace(partial, self.path)

    def _key(self, record: dict) -> tuple:
        return tuple(record.get(k) for k in self.KEY)


def _split(queries: list[dict], split: str, limit: int) -> list[dict]:
    chosen = [q for q in queries if q["split"] == split]
    return chosen[:limit] if limit else chosen


def _right(graph: SemanticGraph, query: dict) -> set[str]:
    """Ids of the objects that answer a command, by their true label."""
    return {o.id for o in graph.objects.values() if o.label in query["answers"]}


def _calibrate(graph: SemanticGraph, calibration: list[dict], alpha: float, method: str) -> float:
    graph.config.alpha, graph.config.set_method = alpha, method
    return graph.calibrate([(q["text"], _right(graph, q)) for q in calibration])


def _escalation(graph: SemanticGraph, query: dict, repeat: int) -> dict:
    """CLIP's top pick and the VLM's pick over the calibrated set, to price any escalation size."""
    kept = graph.candidates(graph.parse(query["text"]))
    before = len(graph.vlm.usage)
    start = time.perf_counter()
    chosen = graph.verify(query["text"], kept[: graph.config.max_candidates]) if len(kept) > 1 else None
    spent = graph.vlm.usage[before:]
    return {
        "experiment": "escalation", "repeat": repeat, "text": query["text"], "split": query["split"],
        "type": query["type"], "right": sorted(_right(graph, query)), "set_size": len(kept),
        "clip_pick": kept[0].obj.id if kept else "", "vlm_pick": chosen.obj.id if chosen else "",
        "vlm_tokens": sum(u["prompt_eval_count"] + u["eval_count"] for u in spent),
        "vlm_seconds": time.perf_counter() - start if spent else 0.0,
    }


def _tune(rows: list[dict]) -> int:
    """Largest set CLIP answers alone: best accuracy, then fewest VLM calls."""
    def accuracy(size: int) -> float:
        picks = [r["clip_pick"] if r["set_size"] <= size else r["vlm_pick"] for r in rows]
        return sum(p in r["right"] for p, r in zip(picks, rows)) / len(rows)

    largest = max(r["set_size"] for r in rows)
    return max(range(1, largest + 1), key=lambda size: (accuracy(size), size))


def _record(experiment: str, method: str, graph: SemanticGraph, query: dict, result, repeat: int,
            right: set[str] | None = None, **extra) -> dict:
    return {
        "experiment": experiment, "method": method, "repeat": repeat,
        "text": query["text"], "type": query["type"], "group": query["group"],
        "right": sorted(right if right is not None else _right(graph, query)),
        **asdict(result), **extra,
    }


def _parse(graph: SemanticGraph, llm: OllamaLLM, query: dict, repeat: int) -> dict:
    start, before = time.perf_counter(), len(llm.usage)
    parsed = graph.parse(query["text"])
    spent = llm.usage[before:]
    return {
        "experiment": "parse", "repeat": repeat, "text": query["text"], "group": query["group"],
        "parsed": asdict(parsed), "seconds": time.perf_counter() - start,
        "tokens_in": sum(u["prompt_eval_count"] for u in spent),
        "tokens_out": sum(u["eval_count"] for u in spent),
    }


def _noise(log, graph, snapshot, calibration, test, share) -> None:
    """A, B and D when a share of the labels is swapped for another of the same category."""
    rng = np.random.default_rng(0)
    noisy = _graph_like(graph, snapshot, copy.deepcopy(graph.objects))
    objects = list(noisy.objects.values())
    for obj in rng.choice(objects, round(share * len(objects)), replace=False):
        peers = [o.label for o in objects if o.category == obj.category and o.label != obj.label]
        obj.label = str(rng.choice(peers or [o.label for o in objects if o.label != obj.label]))
    truth = {q["text"]: _right(graph, q) for q in calibration + test}
    noisy.config.alpha, noisy.config.set_method = MAIN_ALPHA, MAIN_SET
    noisy.calibrate([(q["text"], truth[q["text"]]) for q in calibration])
    methods = [("A", lambda text: full_graph(noisy, graph.llm, text)),
               ("B", lambda text: clip_only(noisy, text)),
               ("D", lambda text: cascade(noisy, graph.vlm, text))]
    for name, run in methods:
        for query in test:
            if not log.get("noise", name, 0, query["text"]):
                log.add(_record("noise", name, noisy, query, run(query["text"]), 0, truth[query["text"]]))


def _scaling(log, graph, snapshot, calibration, test, scales) -> None:
    """Cost against graph size, padding the house with near-copies of its objects."""
    wide = OllamaLLM(PARSER, context=16384)
    for size in scales:
        rng = np.random.default_rng(size)
        big = _graph_like(graph, snapshot, _padded(graph, size, rng))
        big.config.alpha, big.config.set_method = MAIN_ALPHA, MAIN_SET
        big.calibrate([(q["text"], _right(big, q)) for q in calibration])
        for query in test:
            if log.get("scaling", None, None, query["text"], size=size):
                continue
            parsed = big.parse(query["text"])
            start = time.perf_counter()
            big.select(parsed, len(big.objects))
            select_s = time.perf_counter() - start
            row = {"experiment": "scaling", "size": size, "text": query["text"],
                   "select_seconds": select_s, "set_size": len(big.candidates(parsed))}
            if size <= FULL_GRAPH_MAX:
                result = full_graph(big, wide, query["text"])
                row.update(full_graph_tokens=result.tokens_in, full_graph_seconds=result.seconds)
            log.add(row)
        if not log.get("spatial_index", None, None, None, size=size):
            log.add({"experiment": "spatial_index", "size": size, **_spatial(big, rng)})


def _padded(graph: SemanticGraph, size: int, rng) -> dict[str, ObjectInstance]:
    """The real objects plus near-copies at random spots of random rooms."""
    objects = copy.deepcopy(graph.objects)
    rooms = [r for r in graph.rooms.values() if r.polygon]
    originals = list(graph.objects.values())
    while len(objects) < size:
        source, room = originals[rng.integers(len(originals))], rooms[rng.integers(len(rooms))]
        xs, ys = zip(*room.polygon)
        x, y = rng.uniform(min(xs), max(xs)), rng.uniform(min(ys), max(ys))
        where = graph.area_for_point(x, y, source.position.z)
        if where.area != room.name:
            continue
        vector = source.embedding + rng.normal(0, 0.02, source.embedding.shape)
        clone = copy.copy(source)
        clone.id, clone.position = f"obj_{len(objects) + 1:05d}", Point(x, y, source.position.z)
        clone.area, clone.sublocation = where.area, where.sublocation
        clone.embedding = (vector / np.linalg.norm(vector)).astype(np.float32)
        objects[clone.id] = clone
    return objects


def _spatial(graph: SemanticGraph, rng) -> dict:
    """The "next to" filter: brute force as the core does it, against a KD-tree."""
    near = graph.config.near_m
    points = np.array([(o.position.x, o.position.y) for o in graph.objects.values()])
    anchors = points[rng.integers(len(points), size=200)]
    start = time.perf_counter()
    brute = [[i for i, p in enumerate(points) if math.hypot(*(p - a)) <= near] for a in anchors]
    brute_s = time.perf_counter() - start
    start = time.perf_counter()
    tree = cKDTree(points)
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    found = tree.query_ball_point(anchors, near)
    tree_s = time.perf_counter() - start
    same = all(sorted(a) == sorted(b) for a, b in zip(brute, found))
    return {"anchors": len(anchors), "brute_seconds": brute_s, "tree_build_seconds": build_s,
            "tree_seconds": tree_s, "same_result": same}


def _graph_like(graph: SemanticGraph, snapshot: Snapshot, objects: dict) -> SemanticGraph:
    """Same house, models and parses, different objects."""
    twin = SemanticGraph.from_snapshot(
        Snapshot(rooms=snapshot.rooms, objects=objects, t_capture=snapshot.t_capture,
                 map_name=snapshot.map_name, embedding_dim=snapshot.embedding_dim),
        config=copy.deepcopy(graph.config), encoder=graph.encoder, llm=graph.llm, vlm=graph.vlm,
    )
    twin.parse_cache = dict(graph.parse_cache)
    return twin


def _warm(model: OllamaLLM) -> None:
    """Load the model before timing anything, and forget the call."""
    model.complete_json("Say ok.", {"type": "object", "properties": {"ok": {"type": "string"}}})
    model.usage.pop()


def _ids(matches) -> list[tuple[str, float]]:
    return [(m.obj.id, m.score) for m in matches]


def _environment(args, deterministic: bool | None) -> dict:
    base = "http://localhost:11434"
    models = requests.get(f"{base}/api/tags", timeout=5).json()["models"]
    return {
        "args": vars(args),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "commit": _run("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"),
        "gpu": _run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
        "power_mode": _run("nvpmodel", "-q"),
        "ollama": requests.get(f"{base}/api/version", timeout=5).json()["version"],
        "models": {m["name"]: m["digest"][:12] for m in models if m["name"] in (PARSER, VERIFIER)},
        "loaded": {m["name"]: m["size_vram"] for m in requests.get(f"{base}/api/ps", timeout=5).json()["models"]},
        "temperature": 0.0,
        "seed": 0,
        "select_deterministic": deterministic,
    }


def _run(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


if __name__ == "__main__":
    main()
