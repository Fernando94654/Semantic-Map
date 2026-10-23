"""Offline grounding check: where the VLM looks when it picks a photo from the grid.

Runs the Hugging Face twin of the Ollama verifier with eager attention, over the
same numbered grid the cascade sends, for the top-k of every test command. Needs
the GPU free of Ollama models and a results file for the parses.

    python -m eval.grounding [results/<host>.json]
"""

from __future__ import annotations

import io
import json
import socket
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from adapters.clip_encoder import ClipEncoder
from adapters.files import load_snapshot
from adapters.llm import COLUMNS, TILE, grid
from core.config import SemanticMapConfig
from core.graph import VERIFY_PROMPT, SemanticGraph
from core.types import ParsedQuery
from main import MOCK_SNAPSHOT

ROOT = Path(__file__).resolve().parent.parent
# The Hugging Face twin of qwen2.5vl:7b, in 4 bits to fit 8 GB with eager attention.
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / f"{socket.gethostname()}.json"
    records = json.loads(path.read_text())["records"]
    parses = {r["text"]: r["parsed"] for r in records if r["experiment"] == "parse" and r["repeat"] == 0}
    tests = {r["text"]: r["right"] for r in records if r["experiment"] == "main" and r["method"] == "B"}

    config = SemanticMapConfig()
    graph = SemanticGraph.from_snapshot(load_snapshot(MOCK_SNAPSHOT, config), config=config, encoder=ClipEncoder())
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
    )
    numbers = [processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 10)]

    rows = []
    for text, right in tests.items():
        shown = graph.select(ParsedQuery(**parses[text]), gated=False)
        if len(shown) < 2:
            continue
        mass, probabilities = _look(model, processor, numbers, text, [m.obj.crop for m in shown])
        answer, looked = int(probabilities.argmax()), int(mass.argmax())
        correct = [i for i, m in enumerate(shown) if m.obj.id in right]
        rows.append({
            "text": text, "shown": [m.obj.id for m in shown], "correct": correct,
            "answer": answer, "looked": looked, "mass": mass.tolist(),
            "probabilities": probabilities.tolist(),
        })

    graded = [r for r in rows if r["correct"]]
    summary = {
        "model": MODEL, "commands": len(rows), "with_a_right_photo": len(graded),
        "answer_accuracy": _share(r["answer"] in r["correct"] for r in graded),
        "pointing_accuracy": _share(r["looked"] in r["correct"] for r in graded),
        "attention_answer_agreement": _share(r["looked"] == r["answer"] for r in rows),
        "pointing_when_right": _share(r["looked"] in r["correct"] for r in graded if r["answer"] in r["correct"]),
        "pointing_when_wrong": _share(r["looked"] in r["correct"] for r in graded if r["answer"] not in r["correct"]),
    }
    out = path.with_name(path.stem + "_grounding.json")
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"-> {out.relative_to(ROOT)}")


def _look(model, processor, numbers, text, crops):
    """Attention of the answer position over each grid cell, and the answer's distribution."""
    image = Image.open(io.BytesIO(grid(crops)))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": VERIFY_PROMPT.format(n=len(crops), text=text) + " Answer with the number only."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
    logits = output.logits[0, -1, numbers[: len(crops)]].float()
    # Last position, averaged over heads and over the second half of the layers.
    layers = output.attentions[len(output.attentions) // 2:]
    attention = torch.stack([a[0, :, -1].float().mean(0) for a in layers]).mean(0)
    visual = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().squeeze(1)
    _, height, width = inputs["image_grid_thw"][0].tolist()
    merge = processor.image_processor.merge_size
    token_px = processor.image_processor.patch_size * merge
    rows, cols = height // merge, width // merge
    assert len(visual) == rows * cols, "unexpected visual token layout"
    scale_x, scale_y = image.width / (cols * token_px), image.height / (rows * token_px)
    columns = min(len(crops), COLUMNS)
    mass = torch.zeros(len(crops))
    for k, position in enumerate(visual.tolist()):
        r, c = divmod(k, cols)
        x, y = (c + 0.5) * token_px * scale_x, (r + 0.5) * token_px * scale_y
        cell = int(y // TILE) * columns + int(x // TILE)
        if cell < len(crops):
            mass[cell] += attention[position].item()
    return mass / mass.sum(), torch.softmax(logits, 0).cpu()


def _share(flags) -> float:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else float("nan")


if __name__ == "__main__":
    main()
