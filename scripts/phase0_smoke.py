"""Phase 0 deliverable 1: can the chosen model load and forward on this machine?

Loads the config named on the command line through the repo's own adapter
(so the smoke test exercises exactly the code path every experiment uses),
runs a short capture + greedy generation on one FLORES sentence, and merges
what happened into results/phase0-feasibility.json.

Run: uv run python -m scripts.phase0_smoke qwen3-8b
"""

import json
import sys
import time
from pathlib import Path

import torch

from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

RESULTS = Path("results/phase0-feasibility.json")

def merge(section: str, payload: dict) -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[section] = payload
    RESULTS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

def available_gib() -> float:
    fields = dict(
        line.split(":") for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
    )
    return int(fields["MemAvailable"].strip().split()[0]) / 1024**2

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    before = available_gib()
    start = time.time()
    adapter = load_adapter(config)
    loaded = time.time() - start
    cfg = adapter.cfg

    spanish, english = load_pairs(str(default_pairs_path("flores-es-en-dev")), limit=1)[0]
    prompt = translation_prompt(spanish, form="instruction")

    start = time.time()
    activations = adapter.capture([prompt])
    capture_time = time.time() - start
    start = time.time()
    completion = adapter.generate([prompt], max_new_tokens=48)[0]
    generate_time = time.time() - start

    merge("model_smoke", {
        "config": config,
        "hf_name": cfg.hf_name,
        "dtype": cfg.dtype,
        "device": cfg.device,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_heads": cfg.n_heads,
        "load_seconds": round(loaded, 1),
        "capture_shape": list(activations.shape),
        "capture_seconds": round(capture_time, 2),
        "generate_seconds_48_tokens": round(generate_time, 2),
        "mem_available_gib_before": round(before, 1),
        "mem_available_gib_after": round(available_gib(), 1),
        "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 1) if torch.cuda.is_available() else None,
        "prompt": prompt,
        "completion": completion,
        "reference": english,
        "command": f"uv run python -m scripts.phase0_smoke {config}",
    })
    print(json.dumps(json.loads(RESULTS.read_text())["model_smoke"], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
