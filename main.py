"""Entry point for offline work: wires the graph to the local adapters.

On the robot the same SemanticGraph is built in a ROS node with ROS adapters.
"""

from __future__ import annotations

from adapters.clip_encoder import ClipEncoder
from adapters.clock import SystemClock
from adapters.llm import OllamaLLM
from core.config import SemanticMapConfig
from core.graph import SemanticGraph


class LocalSemanticMap:
    """SemanticGraph with CLIP, Ollama and the wall clock running in-process."""

    def __init__(self, config: SemanticMapConfig | None = None) -> None:
        self.config = config or SemanticMapConfig()
        self.encoder = ClipEncoder()
        self.llm = OllamaLLM()
        self.clock = SystemClock()
        self.graph = SemanticGraph(
            config=self.config,
            encoder=self.encoder,
            llm=self.llm,
            clock=self.clock,
        )


def main() -> None:
    LocalSemanticMap()


if __name__ == "__main__":
    main()
