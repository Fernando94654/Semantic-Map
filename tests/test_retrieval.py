"""Retrieval with real CLIP over the photos, and command resolution with Ollama.

Skipped when CLIP or the Ollama server is missing.
"""

import json

import pytest
import requests

from core.graph import SemanticGraph
from core.types import ParsedQuery

pytest.importorskip("clip")

from adapters.clip_encoder import ClipEncoder  # noqa: E402
from adapters.llm import OllamaLLM  # noqa: E402
from main import MOCK_SNAPSHOT  # noqa: E402
from scripts.embed_images import embed_images  # noqa: E402

QUERIES = json.loads((MOCK_SNAPSHOT.parent / "queries.json").read_text())


@pytest.fixture(scope="module")
def encoder():
    return ClipEncoder()


@pytest.fixture(scope="module")
def image_vectors(encoder):
    return embed_images(encoder)


@pytest.fixture(scope="module")
def llm():
    llm = OllamaLLM()
    try:
        models = requests.get(f"{llm.base_url}/api/tags", timeout=2).json()["models"]
    except requests.RequestException:
        pytest.skip("Ollama is not running")
    if llm.model not in {m["name"] for m in models}:
        pytest.skip(f"{llm.model} is not pulled")
    return llm


@pytest.fixture
def graph(snapshot, config, encoder, image_vectors):
    """The mock graph with every object embedded from its photo."""
    for object_id, vector in image_vectors.items():
        snapshot.objects[object_id].embedding = vector
    return SemanticGraph.from_snapshot(snapshot, config=config, encoder=encoder)


def labels(matches):
    return [m.obj.label for m in matches]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("red can", "coca_cola"),
        ("blue can", "pepsi"),
        ("white cup", "cup"),
        ("black mug", "mug"),
        ("orange fruit", "orange"),
        ("banana", "banana"),
        ("cleaning product", "dish_soap"),
        ("milk", "milk"),
        ("cereal box", "cereal"),
    ],
)
def test_find(graph, query, expected):
    assert graph.find(query)[0].obj.label == expected


def top1_accuracy(graph, name_weight):
    graph.config.name_weight = name_weight
    hits = [graph.find(query, 1) for query in QUERIES]
    return sum(bool(h) and h[0].obj.label in QUERIES[q] for q, h in zip(QUERIES, hits))


def test_mixing_name_and_appearance_beats_either_alone(graph):
    mixed = top1_accuracy(graph, 0.5)
    assert mixed >= top1_accuracy(graph, 0.0) and mixed >= top1_accuracy(graph, 1.0)


@pytest.mark.xfail(reason="the name pulls 'a drink' towards the cup, and the milk carton fails sim_min")
def test_find_a_drink(graph):
    drinks = {"coca_cola", "pepsi", "milk", "orange_juice", "water_bottle"}
    assert graph.find("a drink")[0].obj.label in drinks


def test_unrelated_query_finds_nothing(graph):
    assert graph.find("elephant") == []


def test_room_restricts_candidates(graph):
    matches = graph.select(ParsedQuery(target="drink", room="kitchen"))
    assert all(m.obj.area == "kitchen" for m in matches)
    assert matches[0].obj.label in {"coca_cola", "pepsi", "orange_juice", "milk"}


def test_surface_anchor_returns_what_is_on_it(graph):
    matches = graph.select(ParsedQuery(anchor="bed", relation="next_to"))
    assert set(labels(matches)) == {"banana", "book"}


def test_object_anchor_is_matched_by_name(graph):
    matches = graph.select(ParsedQuery(anchor="coca cola", relation="next_to"))
    assert set(labels(matches)) == {"pepsi", "milk"}


def test_anchor_then_target(graph):
    matches = graph.select(ParsedQuery(target="fruit", anchor="counter", relation="on"))
    assert set(labels(matches)[:2]) == {"orange", "apple"}


def test_parse_extracts_room_in_english(graph, llm):
    graph.llm = llm
    parsed = graph.parse("tráeme la bebida de la cocina")
    assert parsed.room == "kitchen" and parsed.anchor == ""


def test_resolve_relational_command(graph, llm):
    graph.llm = llm
    assert set(labels(graph.resolve("lo que está junto a la cama"))) == {"banana", "book"}
