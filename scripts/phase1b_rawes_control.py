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

A common pipe could be: conditions | mean_abs_activation | contrast | flag | survival | write

Run: uv run python -m scripts.phase1b_rawes_control qwen3-8b 5000
"""

import json
import sys

from scripts.phase1a_neuron_scan import DEFAULT_TOKENS, TOP
from scripts.phase1a_neuron_scan import RESULTS as PHASE1A_RESULTS
from src.data.translation import default_pairs_path, load_pairs
from src.experiment import translation_study as study
from src.methods import neurons
from src.model.adapter import load_adapter
from src.telemetry.results import guard, result

RESULTS = result("phase1a-rawes-control.json")

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOKENS
    guard(config)
    if not PHASE1A_RESULTS.exists():
        raise SystemExit(f"{PHASE1A_RESULTS} does not exist; run scripts.phase1a_neuron_scan first")

    adapter = load_adapter(config)
    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")))
    conditions = {
        "raw_es": [spanish for spanish, _ in flores],
        "en_only": [english for _, english in flores],
        "control": neurons.control_texts(len(flores)),
    }
    means, counted = {}, {}
    for name, texts in conditions.items():
        means[name], counted[name] = neurons.mean_abs_activation(adapter, texts, budget)
        print(name, "tokens:", counted[name], flush=True)

    score = neurons.contrast(means["raw_es"], [means["en_only"], means["control"]])
    flagged, sigma = neurons.flag(score)
    n_layers, d_ff = score.shape
    total = int(flagged.sum())
    spread = neurons.concentration(flagged)
    survivors = neurons.survival(json.loads(PHASE1A_RESULTS.read_text())["top50"], score, TOP)
    in_top = sum(entry["in_second_top"] for entry in survivors)
    still_flagged = sum(entry["flagged_second"] for entry in survivors)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "model": {"config": config, "hf_name": adapter.cfg.hf_name, "dtype": adapter.cfg.dtype,
                  "n_layers": n_layers, "d_ff": d_ff},
        "protocol": "score_rawES = mean|act|(raw Spanish FLORES sources) - (mean|act|(EN refs) + "
                    "mean|act|(math/code))/2, same sites and flagging rule as phase 1a (> 2*sigma)",
        "tokens_per_condition": counted,
        "score_sigma": sigma,
        "flagged_total": total,
        "flagged_per_layer": neurons.per_layer_counts(flagged),
        "concentration": spread,
        "raw_es_top50": neurons.top_neurons(score, TOP),
        "phase1a_top50_survival": {
            "in_raw_es_top50": in_top,
            "flagged_raw_es": still_flagged,
            "entries": survivors,
        },
        "command": f"uv run python -m scripts.phase1b_rawes_control {config} {budget}",
    }, indent=2) + "\n")
    print(f"flagged {total}; bottom {spread['bottom_quarter_flagged']} / top {spread['top_quarter_flagged']}; "
          f"1a-top50 surviving in raw-ES top50: {in_top}, flagged: {still_flagged} -> {RESULTS}")

if __name__ == "__main__":
    main()
