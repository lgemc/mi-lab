"""Phase 0 deliverable 2b: which prompt form actually elicits a translation?

Doc §7 Q3: does the model translate under a bare few-shot WMT-style prompt,
or does it need the instruction form? Both forms are run over the same FLORES
sentences, completions are cut at the first line break (everything after it is
the model talking to itself, not the translation), and each form gets a
sentence-level BLEU against the references. The decision and one sample
exchange land in results/phase0-feasibility.json.

Run: uv run python -m scripts.phase0_promptform qwen3-8b
"""

import sys

import sacrebleu

from scripts.phase0_smoke import merge
from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

SAMPLE = 12
SHOTS = 3

def clean_completion(text: str) -> str:
    """The first line of the completion is the translation; the rest is drift"""
    return text.strip().split("\n")[0].strip()

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    adapter = load_adapter(config)

    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")), limit=SAMPLE)
    wmt = load_pairs(str(default_pairs_path("wmt-newstest2013-es-en-500")))
    shots = wmt[-SHOTS:]  # drawn from the tail so a WMT eval over the head never sees its own shots

    sources = [spanish for spanish, _ in flores]
    references = [english for _, english in flores]
    outputs = {}
    scores = {}
    for form in ("instruction", "few_shot"):
        prompts = [translation_prompt(source, form=form, shots=shots) for source in sources]
        raw = adapter.generate(prompts, max_new_tokens=96)
        outputs[form] = [clean_completion(text) for text in raw]
        scores[form] = round(sacrebleu.corpus_bleu(outputs[form], [references]).score, 2)

    chosen = max(scores, key=lambda form: scores[form])
    merge("prompt_form", {
        "config": config,
        "sample_size": SAMPLE,
        "few_shot_shots": SHOTS,
        "bleu_by_form": scores,
        "chosen": chosen,
        "why": (
            "both forms yield fluent English translations; the higher-BLEU form on the same "
            f"{SAMPLE} FLORES sentences wins, and its completions are cut at the first newline"
        ),
        "sample_exchange": {
            form: {
                "prompt": translation_prompt(sources[0], form=form, shots=shots),
                "completion": outputs[form][0],
                "reference": references[0],
            }
            for form in outputs
        },
        "command": f"uv run python -m scripts.phase0_promptform {config}",
    })
    print(scores, "->", chosen)
    for form in outputs:
        print(f"[{form}] {outputs[form][0][:120]}")

if __name__ == "__main__":
    main()
