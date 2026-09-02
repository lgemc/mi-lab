"""Phase 1b deliverable 2: what does the 78-sigma outlier neuron actually do?

Layer 35 neuron 8136 scored 63 (1a, few-shot ES) and 87.8 (raw Spanish) --
an order of magnitude above the field. This pass records its raw per-token
behaviour, together with its five nearest 1a-scored neighbours in the same
layer, across all four conditions: which tokens fire it, how hard, and how
often. `methods.neurons` does the tracing; the traces decide what it
encodes, and the writeup lives beside the numbers in
results/phase1b-outlier-inspection.json.

A common pipe could be: conditions | trace | summarize | write

Run: uv run python -m scripts.phase1b_outlier qwen3-8b
"""

import json
import sys

from src.data.translation import SHOTS, default_pairs_path, load_pairs, translation_prompt
from src.experiment import translation_study as study
from src.methods import neurons
from src.model.adapter import load_adapter
from src.telemetry.results import guard, result

RESULTS = result("phase1b-outlier-inspection.json")

# The outlier and its five highest-1a-score neighbours in the same layer --
# absolute indices, because they are a finding about one checkpoint and not
# a place any other model is expected to have something.
LAYER = 35
NEURONS = [8136, 9490, 11333, 2972, 9307, 3436]
TEXTS_PER_CONDITION = 24

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    guard(config)
    adapter = load_adapter(config)
    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")))[:TEXTS_PER_CONDITION]
    wmt = load_pairs(str(default_pairs_path(study.CORPUS)))
    shots = wmt[-SHOTS:]

    conditions = {
        "es_translation": [translation_prompt(spanish, form="few_shot", shots=shots) for spanish, _ in flores],
        "raw_es": [spanish for spanish, _ in flores],
        "en_only": [english for _, english in flores],
        "control": neurons.control_texts(TEXTS_PER_CONDITION),
    }

    report = {}
    for name, texts in conditions.items():
        traces = neurons.trace(adapter, texts, LAYER, NEURONS)
        report[name] = {str(neuron): neurons.summarize(traces, column) for column, neuron in enumerate(NEURONS)}
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
