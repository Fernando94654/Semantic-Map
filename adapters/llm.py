"""Local LLM over Ollama. Satisfies the LLM port."""

from __future__ import annotations

from typing import Any


class OllamaLLM:
    """Structured completion against an Ollama server."""

    def __init__(
        self,
        model: str = "qwen3:14b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
    ) -> None:
        pass

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Run the prompt and return a dict matching the schema."""
        pass
