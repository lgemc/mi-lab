"""Phase 1b deliverable 1: the raw-Spanish control the 1a caveat asked for.

The 1a ES condition was the full few-shot translation prompt, half of whose
tokens are English -- so its top-only concentration could be Spanish *input*
machinery diluted, or translation-*task* machinery genuinely late in the
stack. One more activation pass separates the two: score the same neurons
with raw Spanish text (bare FLORES source sentences, no shots, no framing)
against the same EN and math/code conditions.

score_rawES = mean|act|(raw Spanish) - (mean|act|(EN) + mean|act|(control))/2

If the 1a top-50 survive as top raw-ES neurons, they are Spanish-language
neurons; if they vanish here, they belong to the translation task context,
not the language.

Run: uv run python -m scripts.phase1b_rawes_control qwen3-8b 5000
"""

import json
import sys
from pathlib import Path

from scripts.phase1a_neuron_scan import RESULTS as PHASE1A_RESULTS
from scripts.phase1a_neuron_scan import control_texts, mean_abs_activation
from src.data.translation import default_pairs_path, load_pairs
from src.model.adapter import load_adapter

RESULTS = Path("results/phase1a-rawes-control.json")

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    adapter = load_adapter(config)
    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")))
    conditions = {
        "raw_es": [spanish for spanish, _ in flores],
        "en_only": [english for _, english in flores],
        "control": control_texts(len(flores)),
    }
    means = {}
    counted = {}
    for name, texts in conditions.items():
        means[name], counted[name] = mean_abs_activation(adapter, texts, budget)
        print(name, "tokens:", counted[name], flush=True)

    score = means["raw_es"] - (means["en_only"] + means["control"]) / 2
    sigma = float(score.std())
    flagged = score > 2 * sigma
    n_layers, d_ff = score.shape
    per_layer = flagged.sum(dim=1)
    quarter = max(1, n_layers // 4)
    total = int(flagged.sum())

    values, order = score.flatten().topk(50)
    top50 = [
        {"layer": int(index // d_ff), "neuron": int(index % d_ff), "score": round(float(value), 6)}
        for value, index in zip(values, order, strict=True)
    ]
    top50_keys = {(entry["layer"], entry["neuron"]) for entry in top50}

    phase1a = json.loads(PHASE1A_RESULTS.read_text())
    survivors = []
    for entry in phase1a["top50"]:
        key = (entry["layer"], entry["neuron"])
        raw_score = float(score[key[0], key[1]])
        survivors.append({
            **entry,
            "raw_es_score": round(raw_score, 6),
            "raw_es_sigmas": round(raw_score / sigma, 2),
            "in_raw_es_top50": key in top50_keys,
            "flagged_raw_es": bool(raw_score > 2 * sigma),
        })

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "model": {"config": config, "hf_name": adapter.cfg.hf_name, "dtype": adapter.cfg.dtype,
                  "n_layers": n_layers, "d_ff": d_ff},
        "protocol": "score_rawES = mean|act|(raw Spanish FLORES sources) - (mean|act|(EN refs) + "
                    "mean|act|(math/code))/2, same sites and flagging rule as phase 1a (> 2*sigma)",
        "tokens_per_condition": counted,
        "score_sigma": sigma,
        "flagged_total": total,
        "flagged_per_layer": {str(layer): int(count) for layer, count in enumerate(per_layer) if int(count)},
        "concentration": {
            "bottom_quarter_flagged": int(per_layer[:quarter].sum()),
            "top_quarter_flagged": int(per_layer[-quarter:].sum()),
            "middle_flagged": total - int(per_layer[:quarter].sum()) - int(per_layer[-quarter:].sum()),
        },
        "raw_es_top50": top50,
        "phase1a_top50_survival": {
            "in_raw_es_top50": sum(entry["in_raw_es_top50"] for entry in survivors),
            "flagged_raw_es": sum(entry["flagged_raw_es"] for entry in survivors),
            "entries": survivors,
        },
        "command": f"uv run python -m scripts.phase1b_rawes_control {config} {budget}",
    }, indent=2) + "\n")
    bottom = int(per_layer[:quarter].sum())
    print(f"flagged {total}; bottom {bottom} / top {int(per_layer[-quarter:].sum())}; "
          f"1a-top50 surviving in raw-ES top50: {sum(entry['in_raw_es_top50'] for entry in survivors)}, "
          f"flagged: {sum(entry['flagged_raw_es'] for entry in survivors)} -> {RESULTS}")

if __name__ == "__main__":
    main()
