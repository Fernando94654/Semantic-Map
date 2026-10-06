"""Local LLM over Ollama. Satisfies the LLM port."""

from __future__ import annotations

import json
from typing import Any

import requests


class OllamaLLM:
    """Structured completion against an Ollama server."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Run the prompt and return a dict matching the schema."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "format": schema,
                "think": False,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        return json.loads(response.json()["message"]["content"])
