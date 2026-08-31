"""Phase 0 deliverable 3: the untouched model's translation quality baseline.

Generates ES->EN translations for the first N FLORES dev pairs under the
prompt form Phase 0 chose (few-shot; see results/phase0-feasibility.json),
then scores them with corpus BLEU (sacrebleu, the fast proxy every ablation
sweep will use) and COMET-22 (Unbabel/wmt22-comet-da, the metric the
pre-registered thresholds in the design doc are written in). COMET wall time
per 100 sentences is recorded because it prices every later sweep (doc §7 Q2).

Every hypothesis is kept in results/phase0-baseline-outputs.json so the two
scores can be recomputed without the GPU.

Run: uv run python -m scripts.phase0_baseline qwen3-8b 200
"""

import gc
import json
import sys
import time
from pathlib import Path

import sacrebleu
import torch

from scripts.phase0_promptform import SHOTS, clean_completion
from scripts.phase0_smoke import merge
from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

OUTPUTS = Path("results/phase0-baseline-outputs.json")

def comet_score(sources, hypotheses, references):
    """COMET-22, returned with its wall time; None with the reason if it cannot run"""
    try:
        start = time.time()
        from comet import download_model, load_from_checkpoint

        model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        data = [
            {"src": source, "mt": hypothesis, "ref": reference}
            for source, hypothesis, reference in zip(sources, hypotheses, references, strict=True)
        ]
        output = model.predict(data, batch_size=16, gpus=1)
        return {
            "system_score": round(float(output.system_score), 4),
            "seconds_total": round(time.time() - start, 1),
            "seconds_per_100": round((time.time() - start) * 100 / len(data), 1),
        }, None
    except Exception as error:  # the fallback is the deliverable, not the crash
        return None, f"{type(error).__name__}: {error}"

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")), limit=size)
    wmt = load_pairs(str(default_pairs_path("wmt-newstest2013-es-en-500")))
    shots = wmt[-SHOTS:]

    sources = [spanish for spanish, _ in flores]
    references = [english for _, english in flores]
    prompts = [translation_prompt(source, form="few_shot", shots=shots) for source in sources]

    # Resumable on purpose: hypotheses are flushed to disk after every slice,
    # so a run killed partway loses one slice, not the afternoon.
    state = {"config": config, "form": "few_shot", "sources": sources, "references": references,
             "hypotheses": [], "generation_seconds": 0.0}
    if OUTPUTS.exists():
        previous = json.loads(OUTPUTS.read_text())
        if previous.get("config") == config and previous.get("sources") == sources:
            state["hypotheses"] = previous["hypotheses"]
            state["generation_seconds"] = previous.get("generation_seconds", 0.0)

    OUTPUTS.parent.mkdir(exist_ok=True)
    if len(state["hypotheses"]) < len(prompts):
        adapter = load_adapter(config)
        slice_size = 4 * adapter.cfg.batch_size
        while len(state["hypotheses"]) < len(prompts):
            done = len(state["hypotheses"])
            start = time.time()
            batch = adapter.generate(prompts[done : done + slice_size], max_new_tokens=96)
            state["hypotheses"].extend(clean_completion(text) for text in batch)
            state["generation_seconds"] = round(state["generation_seconds"] + time.time() - start, 1)
            OUTPUTS.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
            print(f"generated {len(state['hypotheses'])}/{len(prompts)}", flush=True)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()

    hypotheses = state["hypotheses"]
    generation_seconds = state["generation_seconds"]

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    comet, comet_error = comet_score(sources, hypotheses, references)

    merge("metrics_baseline", {
        "config": config,
        "dataset": "flores-es-en-dev",
        "pairs": len(flores),
        "prompt_form": "few_shot",
        "generation_seconds": round(generation_seconds, 1),
        "bleu": round(bleu.score, 2),
        "bleu_signature": str(bleu.format()),
        "chrf": round(chrf.score, 2),
        "comet22": comet,
        "comet22_status": "ok" if comet else f"unavailable ({comet_error}); BLEU is the primary metric",
        "outputs": str(OUTPUTS),
        "command": f"uv run python -m scripts.phase0_baseline {config} {size}",
    })
    print("BLEU", round(bleu.score, 2), "chrF", round(chrf.score, 2), "COMET", comet, comet_error or "")

if __name__ == "__main__":
    main()
