"""Local LLM and VLM over Ollama. Satisfies the LLM and VLM ports."""

from __future__ import annotations

import base64
import io
import json
import math
import time
from typing import Any, Sequence

import requests
from PIL import Image, ImageDraw, ImageFont

# Token counts and nanoseconds Ollama reports for each call.
USAGE_KEYS = (
    "prompt_eval_count", "prompt_eval_cached_count", "eval_count", "total_duration", "load_duration"
)
TILE, COLUMNS = 224, 3
RETRIES, RETRY_WAIT_S = 5, 10


class OllamaLLM:
    """Structured completion against an Ollama server, optionally over images."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 0,
        context: int = 8192,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.seed = seed
        self.context = context
        # One entry per call, with the USAGE_KEYS.
        self.usage: list[dict[str, int]] = []

    def complete_json(
        self, prompt: str, schema: dict[str, Any], images: Sequence[str] = ()
    ) -> dict[str, Any]:
        """Run the prompt, with the image files as one numbered grid, and return the schema's dict."""
        message = {"role": "user", "content": prompt}
        if images:
            message["images"] = [base64.b64encode(grid(images)).decode()]
        request = {
            "model": self.model,
            "messages": [message],
            "format": schema,
            "think": False,
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.context},
        }
        # A crashed runner answers 500 or drops the connection; Ollama restarts it.
        for attempt in range(RETRIES):
            try:
                response = requests.post(f"{self.base_url}/api/chat", json=request, timeout=600)
            except requests.ConnectionError:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(RETRY_WAIT_S)
                continue
            if response.status_code < 500:
                break
            time.sleep(RETRY_WAIT_S)
        response.raise_for_status()
        body = response.json()
        self.usage.append({k: body.get(k, 0) for k in USAGE_KEYS})
        return json.loads(body["message"]["content"])


def grid(paths: Sequence[str]) -> bytes:
    """Tile the images in a JPEG, each with its 1-based number in a yellow box."""
    columns = min(len(paths), COLUMNS)
    sheet = Image.new("RGB", (columns * TILE, math.ceil(len(paths) / columns) * TILE), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=40)
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((TILE - 8, TILE - 8))
        x, y = (i % columns) * TILE + 4, (i // columns) * TILE + 4
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + 48, y + 48], fill="yellow")
        draw.text((x + 12, y + 2), str(i + 1), fill="black", font=font)
    buffer = io.BytesIO()
    sheet.save(buffer, "JPEG")
    return buffer.getvalue()
