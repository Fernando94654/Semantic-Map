"""Offline entry point: the mock snapshot with CLIP and Ollama running locally.

    python main.py "what is next to the bed" "tráeme la bebida de la cocina"
"""

from __future__ import annotations

import sys
from pathlib import Path

from adapters.clip_encoder import ClipEncoder
from adapters.files import load_snapshot
from adapters.llm import OllamaLLM
from core.config import SemanticMapConfig
from core.graph import SemanticGraph

MOCK_SNAPSHOT = Path(__file__).parent / "data" / "mock_snapshot.json"


def build_graph(path: Path = MOCK_SNAPSHOT) -> SemanticGraph:
    config = SemanticMapConfig()
    return SemanticGraph.from_snapshot(
        load_snapshot(path, config), config=config, encoder=ClipEncoder(), llm=OllamaLLM()
    )


def main() -> None:
    graph = build_graph()
    for query in sys.argv[1:]:
        parsed = graph.parse(query)
        print(f"{query}\n  {parsed}")
        for match in graph.select(parsed):
            obj = match.obj
            print(f"  {match.score:.2f}  {obj.id}  {obj.label}  {obj.area}/{obj.sublocation}")


if __name__ == "__main__":
    main()
