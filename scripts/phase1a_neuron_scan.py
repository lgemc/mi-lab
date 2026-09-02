"""Phase 1a: the cross-lingual FFN neuron scan (Tang et al., 2402.16438).

Three conditions through the same model: ES->EN translation prompts (the
few-shot form Phase 0 chose), EN-only text (the FLORES English references),
and a non-linguistic control (synthetic arithmetic and code, generated
locally so the condition owes nothing to a download). `methods.neurons` does
the measuring and the flagging; this script names the conditions and writes
the report.

score(neuron) = ES_translation - (EN + control) / 2, in mean-|activation|
units. Neurons with score > 2 sigma of the score distribution are flagged,
grouped by layer, and summarized bottom-vs-top: Tang et al. predict the
language-specific machinery concentrates in the first and last quarters of
the stack.

A common pipe could be: conditions | mean_abs_activation | contrast | flag | concentration | write

Run: uv run python -m scripts.phase1a_neuron_scan qwen3-8b 5000
"""

import json
import sys

from src.data.translation import SHOTS, default_pairs_path, load_pairs, translation_prompt
from src.experiment import translation_study as study
from src.methods import neurons
from src.model.adapter import load_adapter
from src.telemetry.results import guard, result

DEFAULT_TOKENS = 5000    # real tokens per condition; enough for a stable mean over d_ff neurons
TOP = 50

RESULTS = result("phase1a-neuron-scan.json")

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TOKENS
    guard(config)

    adapter = load_adapter(config)
    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")))
    wmt = load_pairs(str(default_pairs_path(study.CORPUS)))
    shots = wmt[-SHOTS:]

    conditions = {
        "es_translation": [translation_prompt(spanish, form="few_shot", shots=shots) for spanish, _ in flores],
        "en_only": [english for _, english in flores],
        "control": neurons.control_texts(len(flores)),
    }

    means, counted = {}, {}
    for name, texts in conditions.items():
        means[name], counted[name] = neurons.mean_abs_activation(adapter, texts, budget)
        print(name, "tokens:", counted[name])

    score = neurons.contrast(means["es_translation"], [means["en_only"], means["control"]])
    flagged, sigma = neurons.flag(score)
    n_layers, d_ff = score.shape
    total = int(flagged.sum())
    spread = neurons.concentration(flagged)

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps({
        "model": {"config": config, "hf_name": adapter.cfg.hf_name, "dtype": adapter.cfg.dtype,
                  "n_layers": n_layers, "d_ff": d_ff},
        "protocol": "Tang et al. 2402.16438: score = mean|act|(ES) - (mean|act|(EN) + mean|act|(control))/2, "
                    "measured at each MLP down-projection input over real (non-padding) tokens; flag score > 2*sigma",
        "tokens_per_condition": counted,
        "score_sigma": sigma,
        "score_mean": float(score.mean()),
        "flagged_total": total,
        "flagged_share": round(total / (n_layers * d_ff), 6),
        "flagged_per_layer": neurons.per_layer_counts(flagged),
        "concentration": spread,
        "top50": neurons.top_neurons(score, TOP),
        "command": f"uv run python -m scripts.phase1a_neuron_scan {config} {budget}",
    }, indent=2) + "\n")
    print(f"flagged {total} neurons; bottom {spread['bottom_quarter_flagged']} / "
          f"top {spread['top_quarter_flagged']} of {n_layers} layers -> {RESULTS}")

if __name__ == "__main__":
    main()
