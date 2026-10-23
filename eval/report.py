"""Tables and figures from a results file.

    python -m eval.report [results/<host>.json]
"""

from __future__ import annotations

import json
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# Categorical slots in fixed order, one per method.
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#6250d6"}
MARKERS = {"A": "s", "B": "^", "C": "D", "D": "o"}
NAMES = {"A": "A full graph", "B": "B CLIP only", "C": "C VLM always", "D": "D cascade"}
INK, MUTED = "#0b0b0b", "#8a8984"
MAIN_ALPHA, MAIN_SET = 0.1, "aps"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / f"{socket.gethostname()}.json"
    data = json.loads(path.read_text())
    records = data["records"]
    parses = {(r["repeat"], r["text"]): r for r in records if r["experiment"] == "parse"}
    main_rows = [r for r in records if r["experiment"] == "main"]
    runs = _runs(main_rows)
    chosen = {m: runs[m, MAIN_ALPHA, MAIN_SET] if m == "D" else runs[m, None, None] for m in "ABCD"}
    escalation = [r for r in records if r["experiment"] == "escalation" and r["split"] == "test"]
    tuned = Counter(r["escalate_above"] for r in chosen["D"]).most_common(1)[0][0]

    lines = [f"# Results ({data['environment']['host']}, commit {data['environment']['commit']})", ""]
    lines += _table(
        f"Main comparison (test split, D: APS set at alpha {MAIN_ALPHA}, escalate above {tuned})",
        ["method", "accuracy", "error", "ask", "coverage", "completeness", "tokens/query",
         "VLM calls/query", "latency p50 (s)", "repeat agreement", "paraphrase agreement"],
        [[NAMES[m], *_summary(chosen[m], parses)] for m in "ABCD"],
    )
    lines += _table(
        "Accuracy by command type",
        ["method", *sorted({r["type"] for r in main_rows})],
        [[NAMES[m], *[_pct(_accuracy([r for r in chosen[m] if r["type"] == kind]))
                      for kind in sorted({r["type"] for r in main_rows})]]
         for m in "ABCD"],
    )
    lines += _table(
        "Where the VLM helps: commands with more than one candidate in the set",
        ["type", "commands", "CLIP top-1", "VLM over the set"],
        [[kind, len(rows), _pct(_hit(rows, "clip_pick")), _pct(_hit(rows, "vlm_pick"))]
         for kind in sorted({r["type"] for r in escalation})
         for rows in [[r for r in escalation if r["type"] == kind and r["set_size"] > 1]]],
    )
    sizes = _routing(escalation)
    lines += _table(
        "Routing: CLIP answers sets up to s, the VLM the rest (test split)",
        ["s", "accuracy", "VLM calls/query", "VLM tokens/query"],
        [[s, _pct(acc), _fmt(calls, 2), _fmt(tokens)] for s, acc, calls, tokens in sizes
         if s == 1 or (acc, calls) != next(c[1:3] for c in sizes if c[0] == s - 1)],
    )
    alphas = sorted({k[1] for k in runs if k[0] == "D"})
    lines += _table(
        "Cascade against alpha: score threshold against APS mass",
        ["alpha", "set", "target coverage", "coverage", "accuracy", "error", "ask", "mean set size",
         "VLM calls/query", "tokens/query"],
        [[a, method, _pct(1 - a), *_sweep(runs["D", a, method], parses)]
         for a in alphas for method in ("threshold", "aps")],
    )
    noise = [r for r in records if r["experiment"] == "noise"]
    lines += _table(
        "With 25% of the labels swapped",
        ["method", "accuracy", "error", "ask"],
        [[NAMES[m], _pct(_accuracy(rows)), _pct(_error(rows)), _pct(_ask(rows))]
         for m in "ABD" for rows in [[r for r in noise if r["method"] == m]]],
    )
    scaling = defaultdict(list)
    for r in records:
        if r["experiment"] == "scaling":
            scaling[r["size"]].append(r)
    spatial = {r["size"]: r for r in records if r["experiment"] == "spatial_index"}
    lines += _table(
        "Scaling with graph size",
        ["objects", "A tokens/query", "A latency (s)", "select latency (ms)", "D mean set size",
         "next_to brute (ms)", "next_to KD-tree (ms)"],
        [[size,
          _fmt(_mean(rows, "full_graph_tokens")), _fmt(_mean(rows, "full_graph_seconds"), 2),
          _fmt(1000 * mean(r["select_seconds"] for r in rows), 2),
          _fmt(mean(r["set_size"] for r in rows), 1),
          _fmt(1000 * spatial[size]["brute_seconds"], 1),
          _fmt(1000 * (spatial[size]["tree_build_seconds"] + spatial[size]["tree_seconds"]), 2)]
         for size, rows in sorted(scaling.items())],
    )
    lines += ["Parse agreement across repeats: " + _pct(_parse_agreement(parses)),
              f"select() deterministic: {data['environment']['select_deterministic']}", ""]

    out = path.parent / "figures"
    out.mkdir(exist_ok=True)
    _routing_figure(sizes, chosen, parses, tuned, out / "routing.png")
    _coverage_figure(runs, alphas, out / "coverage.png")
    _scaling_figure(scaling, runs, parses, out / "tokens_vs_size.png")
    report = path.with_suffix(".md")
    report.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {report.relative_to(ROOT)}, {out.relative_to(ROOT)}/")


def mean(values) -> float:
    """Average that is NaN, not an error, for an empty group."""
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _hit(rows: list[dict], pick: str) -> float:
    return mean(r[pick] in r["right"] for r in rows)


def _routing(rows: list[dict]) -> list[tuple]:
    """Accuracy and VLM cost if CLIP answered every set up to s and the VLM the rest."""
    curve = []
    for s in range(1, max(r["set_size"] for r in rows) + 1):
        escalated = [r["set_size"] > s for r in rows]
        picks = [r["vlm_pick"] if e else r["clip_pick"] for r, e in zip(rows, escalated)]
        curve.append((s, mean(p in r["right"] for p, r in zip(picks, rows)), mean(escalated),
                      mean(r["vlm_tokens"] * e for r, e in zip(rows, escalated))))
    return curve


def _runs(rows: list[dict]) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["method"], r.get("alpha"), r.get("set_method")].append(r)
    return grouped


def _found_right(r: dict) -> bool:
    return r["status"] == "found" and r["picked"] in r["right"]


def _accuracy(rows: list[dict]) -> float:
    return mean(_found_right(r) for r in rows)


def _error(rows: list[dict]) -> float:
    """A wrong object, or nothing when something was there: what asking avoids."""
    wrong = [(r["status"] == "found" and r["picked"] not in r["right"])
             or (r["status"] == "not_found" and bool(r["right"])) for r in rows]
    return mean(wrong)


def _ask(rows: list[dict]) -> float:
    return mean(r["status"] in ("ask", "list") for r in rows)


def _coverage(rows: list[dict]) -> float:
    return mean(bool(set(r["candidates"]) & set(r["right"])) for r in rows)


def _completeness(rows: list[dict]) -> float:
    many = [r for r in rows if len(r["right"]) > 1]
    return mean(len(set(r["candidates"]) & set(r["right"])) / len(r["right"]) for r in many)


def _tokens(rows: list[dict], parses: dict) -> float:
    """Model tokens per command, the parse included for the methods that need it."""
    return mean(r["tokens_in"] + r["tokens_out"] + _parse_cost(r, parses, "tokens") for r in rows)


def _parse_cost(r: dict, parses: dict, what: str) -> float:
    if r["method"] == "A":
        return 0.0
    parse = parses[r["repeat"], r["text"]]
    return parse["seconds"] if what == "seconds" else parse["tokens_in"] + parse["tokens_out"]


def _agreement(rows: list[dict], key) -> float:
    """Share of outcomes equal to the most common one within each group."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append((r["status"], r["picked"]))
    return mean(Counter(g).most_common(1)[0][1] / len(g) for g in groups.values())


def _summary(rows: list[dict], parses: dict) -> list:
    return [
        _pct(_accuracy(rows)), _pct(_error(rows)), _pct(_ask(rows)), _pct(_coverage(rows)),
        _pct(_completeness(rows)), _fmt(_tokens(rows, parses)),
        _fmt(mean(r["vlm_calls"] for r in rows), 2),
        _fmt(median(r["seconds"] + _parse_cost(r, parses, "seconds") for r in rows), 2),
        _pct(_agreement(rows, lambda r: r["text"])),
        _pct(_agreement(rows, lambda r: (r["repeat"], r["group"]))),
    ]


def _sweep(rows: list[dict], parses: dict) -> list:
    return [
        _pct(_coverage(rows)), _pct(_accuracy(rows)), _pct(_error(rows)), _pct(_ask(rows)),
        _fmt(mean(len(r["candidates"]) for r in rows), 1),
        _fmt(mean(r["vlm_calls"] for r in rows), 2), _fmt(_tokens(rows, parses)),
    ]


def _parse_agreement(parses: dict) -> float:
    by_text = defaultdict(list)
    for (_, text), r in parses.items():
        by_text[text].append(json.dumps(r["parsed"], sort_keys=True))
    return mean(Counter(v).most_common(1)[0][1] / len(v) for v in by_text.values())


def _mean(rows: list[dict], key: str):
    values = [r[key] for r in rows if key in r]
    return mean(values) if values else None


def _table(title: str, header: list[str], rows: list[list]) -> list[str]:
    return [f"## {title}", "", "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header), *("| " + " | ".join(map(str, r)) + " |" for r in rows), ""]


def _pct(value: float) -> str:
    return "-" if value != value else f"{100 * value:.1f}%"


def _fmt(value, digits: int = 0) -> str:
    return "-" if value is None or value != value else f"{value:.{digits}f}"


def _axes(title: str, xlabel: str, ylabel: str):
    figure, axes = plt.subplots(figsize=(5, 3.4), dpi=200)
    axes.set_title(title, loc="left", fontsize=10, color=INK)
    axes.set_xlabel(xlabel, fontsize=8, color=INK)
    axes.set_ylabel(ylabel, fontsize=8, color=INK)
    axes.tick_params(labelsize=7, colors=MUTED)
    axes.grid(color="#e6e5e0", linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    return figure, axes


def _routing_figure(curve, chosen, parses, tuned, path: Path) -> None:
    figure, axes = _axes("Accuracy against VLM use", "VLM calls per command", "accuracy (test split)")
    calls, accuracy = [c[2] for c in curve], [c[1] for c in curve]
    axes.plot(calls, accuracy, color=COLORS["D"], linewidth=2, marker="o", markersize=5,
              markeredgecolor="white", zorder=3)
    for s, acc, used, _ in curve:
        if s in (1, tuned, curve[-1][0]):
            axes.annotate(f"s={s}", (used, acc), xytext=(0, 8), textcoords="offset points", fontsize=7,
                          color=INK, ha="center")
    for m in "ABC":
        rows = chosen[m]
        x, y = mean(r["vlm_calls"] for r in rows), _accuracy(rows)
        axes.scatter(x, y, s=40, color=COLORS[m], marker=MARKERS[m], edgecolor="white", linewidth=1.5, zorder=3)
        axes.annotate(NAMES[m], (x, y), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK)
    figure.tight_layout()
    figure.savefig(path)


def _coverage_figure(runs, alphas, path: Path) -> None:
    figure, axes = _axes("Conformal coverage on the test split", "target 1 − α", "empirical coverage")
    targets = [1 - a for a in alphas]
    axes.plot(targets, targets, color=MUTED, linewidth=1, linestyle="--")
    axes.annotate("guarantee", (targets[-1], targets[-1]), xytext=(4, -10), textcoords="offset points",
                  fontsize=7, color=MUTED)
    covered = [_coverage(runs["D", a, MAIN_SET]) for a in alphas]
    axes.plot(targets, covered, color=COLORS["D"], linewidth=2, marker="o", markersize=6,
              markeredgecolor="white")
    axes.annotate(NAMES["D"], (targets[0], covered[0]), xytext=(6, 4), textcoords="offset points",
                  fontsize=7, color=INK)
    figure.tight_layout()
    figure.savefig(path)


def _scaling_figure(scaling, runs, parses, path: Path) -> None:
    figure, axes = _axes("Prompt size against graph size", "objects in the graph", "model tokens per command")
    sizes = [s for s in sorted(scaling) if _mean(scaling[s], "full_graph_tokens") is not None]
    full = [_mean(scaling[s], "full_graph_tokens") for s in sizes]
    axes.plot(sizes, full, color=COLORS["A"], linewidth=2, marker="s", markersize=6, markeredgecolor="white")
    axes.annotate(NAMES["A"], (sizes[-1], full[-1]), xytext=(-60, 6), textcoords="offset points",
                  fontsize=7, color=INK)
    cascade = _tokens(runs["D", MAIN_ALPHA, MAIN_SET], parses)
    every = sorted(scaling)
    axes.plot(every, [cascade] * len(every), color=COLORS["D"], linewidth=2, marker="o", markersize=6,
              markeredgecolor="white")
    axes.annotate(NAMES["D"] + " (measured at 60)", (every[-1], cascade), xytext=(-110, 6),
                  textcoords="offset points", fontsize=7, color=INK)
    axes.set_xscale("log")
    figure.tight_layout()
    figure.savefig(path)


if __name__ == "__main__":
    main()
