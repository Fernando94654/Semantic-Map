"""ROS 2 adapters: the same ports, answered by services on the robot.

The only file that imports rclpy. Calls block until the server answers, so the
node must spin in a MultiThreadedExecutor.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from frida_interfaces.srv import EmbedText, LLMWrapper, MapAreas

SERVICE_TIMEOUT_S = 5.0


def _call(node: Node, srv_type: Any, service: str, request: Any) -> Any:
    client = node.create_client(srv_type, service, callback_group=ReentrantCallbackGroup())
    if not client.wait_for_service(timeout_sec=SERVICE_TIMEOUT_S):
        raise TimeoutError(f"{service} is not available")
    return client.call(request)


class RosClipEncoder:
    """CLIP text embeddings from the vision service."""

    def __init__(self, node: Node, service: str = "/vision/embed_text") -> None:
        self.node = node
        self.service = service

    @property
    def dim(self) -> int:
        return len(self.encode_text("object"))

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a string, L2-normalized."""
        response = _call(self.node, EmbedText, self.service, EmbedText.Request(texts=[text]))
        if not response.success:
            raise RuntimeError(f"{self.service} failed to embed {text!r}")
        vector = np.asarray(response.embeddings, np.float32)
        return vector / np.linalg.norm(vector)


class RosLLM:
    """Structured completion through the HRI LLM wrapper, which only returns text."""

    def __init__(self, node: Node, service: str = "/hri/nlp/llm") -> None:
        self.node = node
        self.service = service

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        question = f"{prompt}\n\nAnswer only with JSON matching this schema:\n{json.dumps(schema)}"
        answer = _call(self.node, LLMWrapper, self.service, LLMWrapper.Request(question=question)).answer
        return json.loads(answer[answer.index("{") : answer.rindex("}") + 1])


class RosClock:
    """The node clock, so simulated time is respected."""

    def __init__(self, node: Node) -> None:
        self.node = node

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds / 1e9


def load_areas(node: Node, service: str = "/navigation/areas_json") -> dict[str, Any]:
    """The active map's areas.json, to pass to SemanticGraph.load_layout."""
    return json.loads(_call(node, MapAreas, service, MapAreas.Request()).areas)
