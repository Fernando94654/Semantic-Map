"""The four methods under comparison. Each takes a command and returns a Result.

The command is parsed once beforehand and cached on the graph, so B, C and D
report only their own cost; the parse is measured separately.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from adapters.llm import OllamaLLM
from core.graph import SemanticGraph

FULL_GRAPH_PROMPT = """A household robot knows these objects, one per line as id: label (room/furniture).
{objects}

Which object does the command ask for? Answer its id, or "" if none fits.

Command: {text}"""


@dataclass
class Result:
    """What a method returned and what it cost."""

    status: str
    picked: str = ""
    candidates: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    vlm_calls: int = 0
    seconds: float = 0.0


def full_graph(graph: SemanticGraph, llm: OllamaLLM, text: str) -> Result:
    """A: the whole graph serialized into the prompt, the LLM picks an id."""
    lines = "\n".join(f"{o.id}: {o.label} ({o.area}/{o.sublocation})" for o in graph.objects.values())
    schema = {
        "type": "object",
        "properties": {"id": {"type": "string", "enum": list(graph.objects) + [""]}},
        "required": ["id"],
    }
    start, before = time.perf_counter(), len(llm.usage)
    picked = llm.complete_json(FULL_GRAPH_PROMPT.format(objects=lines, text=text), schema)["id"]
    result = Result("found" if picked else "not_found", picked, [picked] if picked else [])
    return _charge(result, start, llm, before, "llm_calls")


def clip_only(graph: SemanticGraph, text: str) -> Result:
    """B: the top CLIP match after the graph filter."""
    start = time.perf_counter()
    ranked = graph.select(graph.parse(text))
    candidates = [m.obj.id for m in ranked]
    result = Result("found", candidates[0], candidates) if ranked else Result("not_found")
    result.seconds = time.perf_counter() - start
    return result


def vlm_top_k(graph: SemanticGraph, vlm: OllamaLLM, text: str) -> Result:
    """C: the VLM always chooses among the fixed top-k crops."""
    start, before = time.perf_counter(), len(vlm.usage)
    ranked = graph.select(graph.parse(text))
    if not ranked:
        return Result("not_found", seconds=time.perf_counter() - start)
    chosen = graph.verify(text, ranked)
    result = Result("found" if chosen else "ask", chosen.obj.id if chosen else "",
                    [m.obj.id for m in ranked])
    return _charge(result, start, vlm, before, "vlm_calls")


def cascade(graph: SemanticGraph, vlm: OllamaLLM, text: str) -> Result:
    """D: the calibrated set, with the VLM only when the set holds a tie it can break."""
    start, before = time.perf_counter(), len(vlm.usage)
    answer = graph.answer(text)
    candidates = [m.obj.id for m in graph.candidates(graph.parse(text))]
    picked = answer.matches[0].obj.id if answer.status == "found" else ""
    return _charge(Result(answer.status, picked, candidates), start, vlm, before, "vlm_calls")


def _charge(result: Result, start: float, model: OllamaLLM, before: int, calls: str) -> Result:
    """Add the time and the tokens a model spent since `before`."""
    spent = model.usage[before:]
    result.seconds = time.perf_counter() - start
    result.tokens_in = sum(u["prompt_eval_count"] for u in spent)
    result.tokens_cached = sum(u["prompt_eval_cached_count"] for u in spent)
    result.tokens_out = sum(u["eval_count"] for u in spent)
    setattr(result, calls, len(spent))
    return result
