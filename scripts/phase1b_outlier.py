"""Phase 1b deliverable 2: what does the 78-sigma outlier neuron actually do?

Layer 35 neuron 8136 scored 63 (1a, few-shot ES) and 87.8 (raw Spanish) --
an order of magnitude above the field. This pass records its raw per-token
behaviour, together with its five nearest 1a-scored neighbours in the same
layer, across all four conditions: which tokens fire it, how hard, and how
often. The traces decide what it encodes; the writeup lives beside the
numbers in results/phase1b-outlier-inspection.json.

Run: uv run python -m scripts.phase1b_outlier qwen3-8b
"""

import json
import sys
from pathlib import Path

import torch

from scripts.phase0_promptform import SHOTS
from scripts.phase1a_neuron_scan import control_texts, find_down_projection
from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

RESULTS = Path("results/phase1b-outlier-inspection.json")

LAYER = 35
NEURONS = [8136, 9490, 11333, 2972, 9307, 3436]  # the outlier + its 5 highest-1a-score layer-35 neighbours
TEXTS_PER_CONDITION = 24
TOP_FIRING = 12

def trace_condition(adapter, texts, layer: int, neurons):
    """Per-token activations of the chosen neurons: [(activations [seq, n], token strings)] per text"""
    down = find_down_projection(adapter.mlps[layer], layer)
    grabbed = {}

    def hook(module, args):
        grabbed["value"] = args[0].detach()[..., neurons].float().cpu()

    handle = down.register_forward_pre_hook(hook)
    traces = []
    try:
        for start in range(0, len(texts), adapter.cfg.batch_size):
            batch = texts[start : start + adapter.cfg.batch_size]
            adapter.tokenizer.padding_side = "right"
            encoded = adapter.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            ids = encoded["input_ids"].to(adapter.model.device)
            mask = encoded["attention_mask"].to(adapter.model.device)
            with torch.no_grad():
                adapter.model(ids, attention_mask=mask, use_cache=False)
            for row in range(ids.shape[0]):
                real = int(mask[row].sum())
                tokens = [adapter.tokenizer.decode([token]) for token in ids[row, :real]]
                traces.append((grabbed["value"][row, :real], tokens))
    finally:
        handle.remove()
    return traces

def summarize(traces, column: int):
    """One neuron's behaviour over a condition: moments plus its hardest-firing tokens in context"""
    values = torch.cat([activations[:, column] for activations, _ in traces])
    firing = []
    for activations, tokens in traces:
        for position in range(len(tokens)):
            firing.append((float(activations[position, column]), position, tokens))
    firing.sort(key=lambda item: -abs(item[0]))
    top = [
        {
            "activation": round(value, 2),
            "token": tokens[position],
            "context": "".join(tokens[max(0, position - 5) : position + 1]),
        }
        for value, position, tokens in firing[:TOP_FIRING]
    ]
    above = float((values.abs() > values.abs().mean() + 2 * values.abs().std()).float().mean())
    return {
        "mean_abs": round(float(values.abs().mean()), 3),
        "max_abs": round(float(values.abs().max()), 2),
        "share_tokens_above_2sigma": round(above, 4),
        "top_firing": top,
    }

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    adapter = load_adapter(config)
    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")))
    wmt = load_pairs(str(default_pairs_path("wmt-newstest2013-es-en-500")))
    shots = wmt[-SHOTS:]

    conditions = {
        "es_translation": [
            translation_prompt(spanish, form="few_shot", shots=shots)
            for spanish, _ in flores[:TEXTS_PER_CONDITION]
        ],
        "raw_es": [spanish for spanish, _ in flores[:TEXTS_PER_CONDITION]],
        "en_only": [english for _, english in flores[:TEXTS_PER_CONDITION]],
        "control": control_texts(TEXTS_PER_CONDITION),
    }

    report = {}
    for name, texts in conditions.items():
        traces = trace_condition(adapter, texts, LAYER, NEURONS)
        report[name] = {str(neuron): summarize(traces, column) for column, neuron in enumerate(NEURONS)}
        print(name, "done", flush=True)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "model": {"config": config, "hf_name": adapter.cfg.hf_name, "dtype": adapter.cfg.dtype},
        "layer": LAYER,
        "neurons": NEURONS,
        "texts_per_condition": TEXTS_PER_CONDITION,
        "traces": report,
        "writeup": "filled in after reading the traces",
        "command": f"uv run python -m scripts.phase1b_outlier {config}",
    }, indent=2, ensure_ascii=False) + "\n")
    print("->", RESULTS)

if __name__ == "__main__":
    main()
