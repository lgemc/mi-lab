"""Phase 0 deliverable 3: the untouched model's translation quality baseline.

Generates ES->EN translations for the first N FLORES dev pairs under the
prompt form Phase 0 chose (few-shot; see results/phase0-feasibility.json),
then scores them with corpus BLEU (sacrebleu, the fast proxy every ablation
sweep will use) and COMET-22 (`methods.quality.Comet`, the metric the
pre-registered thresholds in the design doc are written in). COMET wall time
per 100 sentences is recorded because it prices every later sweep (doc §7 Q2).

Every hypothesis is kept in results/phase0-baseline-outputs.json so the two
scores can be recomputed without the GPU, and generation resumes from it.

A common pipe could be: translation_prompt | translate | bleu + chrf + Comet | merge_section

Run: uv run python -m scripts.phase0_baseline qwen3-8b 200
"""

import gc
import sys
import time

import torch

from scripts.phase0_smoke import merge
from src.data.translation import SHOTS, default_pairs_path, load_pairs, translation_prompt
from src.experiment import translation_study as study
from src.methods.knockout import translate
from src.methods.quality import Comet, bleu, bleu_signature, chrf
from src.model.adapter import load_adapter
from src.telemetry.results import guard, load_state, result, save_state

OUTPUTS = result("phase0-baseline-outputs.json")
DEFAULT_SIZE = 200
MAX_NEW_TOKENS = 96
SLICES_PER_SAVE = 4    # generation batches between flushes of the outputs file

def comet_score(sources, hypotheses, references):
    """COMET-22 with its wall time, or None with the reason: the fallback is the deliverable, not the crash"""
    try:
        start = time.time()
        score = Comet().score(sources, hypotheses, references)
        elapsed = time.time() - start
        return {
            "system_score": score,
            "seconds_total": round(elapsed, 1),
            "seconds_per_100": round(elapsed * 100 / len(hypotheses), 1),
        }, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    size = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SIZE
    guard(config)

    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")), limit=size)
    wmt = load_pairs(str(default_pairs_path(study.CORPUS)))
    shots = wmt[-SHOTS:]

    sources = [spanish for spanish, _ in flores]
    references = [english for _, english in flores]
    prompts = [translation_prompt(source, form="few_shot", shots=shots) for source in sources]

    state = {"config": config, "form": "few_shot", "sources": sources, "references": references,
             "hypotheses": [], "generation_seconds": 0.0}
    previous = load_state(OUTPUTS)
    if previous.get("config") == config and previous.get("sources") == sources:
        state["hypotheses"] = previous["hypotheses"]
        state["generation_seconds"] = previous.get("generation_seconds", 0.0)

    if len(state["hypotheses"]) < len(prompts):
        adapter = load_adapter(config)
        slice_size = SLICES_PER_SAVE * adapter.cfg.batch_size
        while len(state["hypotheses"]) < len(prompts):
            done = len(state["hypotheses"])
            start = time.time()
            state["hypotheses"].extend(translate(adapter, prompts[done : done + slice_size], label="baseline",
                                                 max_new_tokens=MAX_NEW_TOKENS))
            state["generation_seconds"] = round(state["generation_seconds"] + time.time() - start, 1)
            save_state(OUTPUTS, state)
            print(f"generated {len(state['hypotheses'])}/{len(prompts)}", flush=True)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()

    hypotheses = state["hypotheses"]
    comet, comet_error = comet_score(sources, hypotheses, references)

    merge("metrics_baseline", {
        "config": config,
        "dataset": "flores-es-en-dev",
        "pairs": len(flores),
        "prompt_form": "few_shot",
        "generation_seconds": round(state["generation_seconds"], 1),
        "bleu": bleu(hypotheses, references),
        "bleu_signature": bleu_signature(hypotheses, references),
        "chrf": chrf(hypotheses, references),
        "comet22": comet,
        "comet22_status": "ok" if comet else f"unavailable ({comet_error}); BLEU is the primary metric",
        "outputs": str(OUTPUTS),
        "command": f"uv run python -m scripts.phase0_baseline {config} {size}",
    })
    print("BLEU", bleu(hypotheses, references), "chrF", chrf(hypotheses, references), "COMET", comet,
          comet_error or "")

if __name__ == "__main__":
    main()
