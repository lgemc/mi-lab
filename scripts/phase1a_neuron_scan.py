"""Phase 1a: the cross-lingual FFN neuron scan (Tang et al., 2402.16438).

Three conditions through the same model: ES->EN translation prompts (the
few-shot form Phase 0 chose), EN-only text (the FLORES English references),
and a non-linguistic control (synthetic arithmetic and code, generated
locally so the condition owes nothing to a download). For every FFN neuron --
the input coordinates of each MLP's down projection, which is where a gated
MLP's post-activation values live -- the mean |activation| over real tokens is
accumulated per condition.

score(neuron) = ES_translation - (EN + control) / 2, in mean-|activation|
units. Neurons with score > 2 sigma of the score distribution are flagged,
grouped by layer, and summarized bottom-vs-top: Tang et al. predict the
language-specific machinery concentrates in the first and last quarters of
the stack.

Run: uv run python -m scripts.phase1a_neuron_scan qwen3-8b 5000
"""

import json
import random
import sys
from pathlib import Path

import torch

from scripts.phase0_promptform import SHOTS
from src.data.translation import default_pairs_path, load_pairs, translation_prompt
from src.model.adapter import load_adapter

RESULTS = Path("results/phase1a-neuron-scan.json")

def control_texts(count: int, seed: int = 0) -> list:
    """Math and code lines with no natural language in them, built locally"""
    rng = random.Random(seed)
    lines = []
    for index in range(count):
        if index % 2 == 0:
            a, b, c = rng.randint(2, 999), rng.randint(2, 999), rng.randint(2, 99)
            lines.append(f"{a} * {b} + {c} = {a * b + c}; ({a} + {b}) / {c} = {(a + b) / c:.4f}")
        else:
            name = rng.choice(["x", "acc", "buf", "val", "tmp"])
            lines.append(
                f"def f{index}({name}):\n    return [{name}[i] ** 2 % {rng.randint(3, 97)} "
                f"for i in range(len({name})) if i % {rng.randint(2, 9)} != 0]"
            )
    return lines

def find_down_projection(mlp, index: int):
    """The linear whose input is the post-activation neuron vector, found by name then by shape"""
    for name in ("down_proj", "c_proj", "dense_4h_to_h", "fc_out", "w2"):
        module = getattr(mlp, name, None)
        if module is not None:
            return module
    raise SystemExit(f"cannot find the down projection of MLP {index} ({type(mlp).__name__}); teach this script")

def mean_abs_activation(adapter, texts, minimum_tokens: int):
    """Mean |activation| per FFN neuron as [layer, d_ff], over at least minimum_tokens real tokens"""
    downs = [find_down_projection(mlp, index) for index, mlp in enumerate(adapter.mlps)]
    sums = None
    tokens = 0
    state = {}

    def make_hook(index: int):
        def hook(module, args):
            value = args[0].detach()
            state["last"][index] = (value.abs().float() * state["mask"][..., None]).sum(dim=(0, 1)).cpu()
        return hook

    handles = [down.register_forward_pre_hook(make_hook(index)) for index, down in enumerate(downs)]
    try:
        for start in range(0, len(texts), adapter.cfg.batch_size):
            batch = texts[start : start + adapter.cfg.batch_size]
            adapter.tokenizer.padding_side = "right"
            encoded = adapter.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            ids = encoded["input_ids"].to(adapter.model.device)
            mask = encoded["attention_mask"].to(adapter.model.device)
            state["mask"] = mask.float()
            state["last"] = {}
            with torch.no_grad():
                adapter.model(ids, attention_mask=mask, use_cache=False)
            stacked = torch.stack([state["last"][index] for index in range(len(downs))])
            sums = stacked.double() if sums is None else sums + stacked.double()
            tokens += int(mask.sum())
            if tokens >= minimum_tokens:
                break
    finally:
        for handle in handles:
            handle.remove()
    return sums / tokens, tokens

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

    adapter = load_adapter(config)
    flores = load_pairs(str(default_pairs_path("flores-es-en-dev")))
    wmt = load_pairs(str(default_pairs_path("wmt-newstest2013-es-en-500")))
    shots = wmt[-SHOTS:]

    # ES condition: full few-shot translation prompts *with* the model's task
    # context; EN condition: bare English sentences; control: math/code.
    conditions = {
        "es_translation": [translation_prompt(spanish, form="few_shot", shots=shots) for spanish, _ in flores],
        "en_only": [english for _, english in flores],
        "control": control_texts(len(flores)),
    }

    means = {}
    counted = {}
    for name, texts in conditions.items():
        means[name], counted[name] = mean_abs_activation(adapter, texts, budget)
        print(name, "tokens:", counted[name])

    score = means["es_translation"] - (means["en_only"] + means["control"]) / 2
    sigma = float(score.std())
    flagged = score > 2 * sigma

    n_layers, d_ff = score.shape
    quarter = max(1, n_layers // 4)
    per_layer = flagged.sum(dim=1)
    total = int(flagged.sum())
    bottom = int(per_layer[:quarter].sum())
    top = int(per_layer[-quarter:].sum())

    values, order = score.flatten().topk(50)
    top50 = [
        {"layer": int(index // d_ff), "neuron": int(index % d_ff), "score": round(float(value), 6)}
        for value, index in zip(values, order, strict=True)
    ]

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
        "flagged_per_layer": {str(layer): int(count) for layer, count in enumerate(per_layer) if int(count)},
        "concentration": {
            "bottom_quarter_layers": quarter,
            "bottom_quarter_flagged": bottom,
            "top_quarter_flagged": top,
            "middle_flagged": total - bottom - top,
            "bottom_plus_top_share": round((bottom + top) / total, 4) if total else None,
        },
        "top50": top50,
        "command": f"uv run python -m scripts.phase1a_neuron_scan {config} {budget}",
    }, indent=2) + "\n")
    print(f"flagged {total} neurons; bottom {bottom} / top {top} of {n_layers} layers -> {RESULTS}")

if __name__ == "__main__":
    main()
