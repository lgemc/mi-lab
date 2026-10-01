"""Phase 2: the activation-difference lens over the SDFT and SFT fine-tunes, against one reference

What each fine-tune adds to the residual stream on average, on text that is
identical for every checkpoint (`src/domains/lm/analysis/actdiff.py` has the
method and its source). It is the residual-stream baseline the dictionaries
have to beat, and the only measurement in phase 2 besides the weight diff that
sees attention.

Per fine-tune and text set, per layer: how large the mean difference is against
the reference's stream, and -- at a few depths -- the vocabulary it promotes and
suppresses under the logit lens. Across fine-tunes, per layer: the cosine
between their mean differences, which says whether SDFT and SFT moved the
stream the same way or different ways.

Run: uv run python -m scripts.phase2_actdiff qwen3-0.6b-sdft qwen3-0.6b-sdft-tooluse qwen3-0.6b-sft-tooluse
"""

import dataclasses
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from src.core.config import load_config
from src.domains.lm.analysis import actdiff
from src.domains.lm.data.sdft import TASKS, task_texts
from src.domains.lm.data.wikitext import passages
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-actdiff.json")
MEANS = result("phase2-actdiff-means.safetensors")
LOG = result("phase2-actdiff.log")
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
TEXT_SOURCE = "Qwen/Qwen3-0.6B"
BUDGET = 2048
MAX_POSITIONS = 3072
DEPTHS = (0.25, 0.5, 0.75, 0.9)


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def sequences(tokenizer, count: int, device) -> Dict[str, list]:
    import torch

    sets = {"wikitext": []}
    for text in passages(count):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"]
        sets["wikitext"].append((ids.to(device), 1))
    for task in TASKS:
        sets[task] = []
        for item in task_texts(SDFT_REPO, TEXT_SOURCE, task, tokenizer, BUDGET):
            prompt = tokenizer(item.prompt, add_special_tokens=False)["input_ids"]
            answer = tokenizer(item.answer, add_special_tokens=False)["input_ids"]
            ids = (prompt + answer)[:MAX_POSITIONS]
            if len(ids) - len(prompt) >= 2:
                sets[task].append((torch.tensor([ids], device=device), len(prompt)))
    return sets


def main(argv: List[str]) -> int:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    configs = [word for word in argv if not word.startswith("--")]
    if len(configs) < 2:
        raise SystemExit("give a reference config and at least one fine-tune")
    pre, posts = configs[0], configs[1:]
    set_log_file(LOG)
    guard(f"{pre}..{'+'.join(posts)}")
    banner("phase 2 activation-difference lens", {
        "pre": pre, "posts": ", ".join(posts), "windows": f"early (positions 1..{actdiff.EARLY}), all",
        "text": f"wikitext, {', '.join(TASKS)} answers of {TEXT_SOURCE}", "artifact": ARTIFACT,
    })
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(load_config(pre).hf_name)
    reference = load(pre)
    device = next(reference.parameters()).device
    sets = sequences(tokenizer, 128, device)
    layers = reference.config.num_hidden_layers
    depth = dataclasses.replace(load_config(pre), n_layers=layers)
    probed = depth.layers(list(DEPTHS))

    report: Dict[str, Any] = {"pre": pre, "posts": posts, "early": actdiff.EARLY, "probed_layers": probed,
                              "models": {}, "cosines": {}}
    means: Dict[str, torch.Tensor] = {}
    for post in posts:
        model = load(post)
        report["models"][post] = {}
        for name, items in sets.items():
            with step(f"{post} on {name} ({len(items)} sequences)") as facts:
                diff = actdiff.mean_difference(reference, model, items, name)
                entry = {"positions": diff.counts}
                for window in diff.sums:
                    mean = diff.mean(window)
                    means[f"{post}|{name}|{window}"] = mean.cpu().contiguous()
                    entry[window] = {
                        "relative_norm": diff.relative_norm(window),
                        "vocabulary": {str(layer): actdiff.top_vocabulary(model, tokenizer, mean[layer])
                                       for layer in probed},
                    }
                report["models"][post][name] = entry
                facts["relative_norm_mid"] = round(entry["all"]["relative_norm"][probed[1]], 4)
        del model
        torch.cuda.empty_cache()
    for first, second in combinations(posts, 2):
        for name in sets:
            for window in ("early", "all"):
                a, b = means[f"{first}|{name}|{window}"], means[f"{second}|{name}|{window}"]
                report["cosines"][f"{first}|{second}|{name}|{window}"] = (
                    torch.nn.functional.cosine_similarity(a, b, dim=-1).tolist())
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=1))
    save_file(means, str(MEANS))
    for post, entry in report["models"].items():
        log(f"{post}: relative norm at layer {probed[1]} (all positions) "
            + ", ".join(f"{name} {entry[name]['all']['relative_norm'][probed[1]]:.3f}" for name in sets))
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
