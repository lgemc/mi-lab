"""Phase 2: what a frozen dictionary can and cannot see of a fine-tune, on the text the fine-tune was about

The repaired form of `phase2_diffing ... dictionary`, which measured one number
-- the fit -- over eight wikitext passages with no interval. Three things change,
each for a reason the proposal's review gave:

- **The text.** Neutral prose stays, as the control. Beside it go the two tasks
  the fine-tunes were trained on, as each eval saw them -- the chat-templated
  prompt -- followed by the base checkpoint's own answer, so every checkpoint is
  read on identical sequences and only the answer positions are scored
  (`src/domains/lm/data/sdft.py` says why the base's answer and not each model's).
- **The question.** Besides how well the dictionary fits each checkpoint, how
  much of the *change* between them it reproduces (`visible_change`), and which
  features fire more or less often, each against a split-half noise floor
  (`src/domains/lm/analysis/features.py`).
- **The uncertainty.** Every per-layer number and every layer mean carries a 95%
  interval from resampling sequences.

The dictionary is named by a config rather than taken from `pre`: the stage
pairs (tool use -> science) diff two fine-tunes under the base checkpoint's
dictionary, which is the only one fitted on anything in this family.

Run: uv run python -m scripts.phase2_features qwen3-0.6b-sdft qwen3-0.6b-sft-tooluse
     uv run python -m scripts.phase2_features qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science --dictionary qwen3-0.6b-sdft
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.config import load_config
from src.domains.lm.analysis import features
from src.domains.lm.data.sdft import TASKS, task_texts
from src.domains.lm.data.wikitext import passages
from src.telemetry.observe import banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-features.json")
LOG = result("phase2-features.log")
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
# The checkpoint whose saved answers make the fixed text, and the eval budget they were generated at.
TEXT_SOURCE = "Qwen/Qwen3-0.6B"
BUDGET = 2048
# Sequences are cut here: past it a position costs a full-width encode per layer per model.
MAX_POSITIONS = 3072
# A feature is "active" at this share of scored positions or above; the Jaccard is stated at it.
FLOOR = 1e-3


class FeaturesRunError(RuntimeError):
    """A run that cannot proceed, said with the argument that would fix it"""


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def sequences(tokenizer, options: Dict[str, Any], device) -> Dict[str, List[Tuple[Any, int]]]:
    """Every text set as (ids, first scored position), identical for both checkpoints"""
    import torch

    sets: Dict[str, List[Tuple[Any, int]]] = {"wikitext": []}
    for text in passages(options["passages"]):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)["input_ids"]
        sets["wikitext"].append((ids.to(device), 1))
    for task in TASKS:
        sets[task] = []
        for item in task_texts(SDFT_REPO, TEXT_SOURCE, task, tokenizer, BUDGET, options["limit"]):
            prompt = tokenizer(item.prompt, add_special_tokens=False)["input_ids"]
            answer = tokenizer(item.answer, add_special_tokens=False)["input_ids"]
            ids = (prompt + answer)[:MAX_POSITIONS]
            if len(ids) - len(prompt) < 2:
                continue
            sets[task].append((torch.tensor([ids], device=device), len(prompt)))
    return sets


def parse(argv: List[str]) -> Dict[str, Any]:
    positional, flags, index = [], {}, 0
    while index < len(argv):
        word = argv[index]
        if word.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else None
            if following is not None and not following.startswith("--"):
                flags[word[2:]] = following
                index += 1
            else:
                flags[word[2:]] = "1"
        else:
            positional.append(word)
        index += 1
    if len(positional) != 2:
        raise FeaturesRunError("give two configs, pre and post")
    return {
        "pre": positional[0],
        "post": positional[1],
        "dictionary": flags.get("dictionary", positional[0]),
        "passages": int(flags.get("passages", 64)),
        "limit": int(flags["limit"]) if "limit" in flags else None,
        "resamples": int(flags.get("resamples", 1000)),
    }


def main(argv: List[str]) -> int:
    from transformers import AutoTokenizer

    options = parse(argv)
    set_log_file(LOG)
    guard(f"{options['pre']}..{options['post']}")
    holder = load_config(options["dictionary"])
    if holder.transcoder is None:
        raise FeaturesRunError(f"'{options['dictionary']}' names no transcoder; pass --dictionary with one that does")
    release = holder.transcoder.release
    banner("phase 2 frozen-dictionary features", {
        "pre": options["pre"], "post": options["post"], "dictionary": f"{release} (from {options['dictionary']})",
        "text": f"wikitext x{options['passages']}, {', '.join(TASKS)} answers of {TEXT_SOURCE}",
        "floor": FLOOR, "artifact": ARTIFACT,
    })
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(load_config(options["pre"]).hf_name)
    with step("loading both checkpoints"):
        pre, post = load(options["pre"]), load(options["post"])
    device = next(pre.parameters()).device
    with step(f"loading {release}") as facts:
        transcoders = features.load_transcoders(release, pre.config.num_hidden_layers, str(device))
        facts["gpu"] = gpu()
    report: Dict[str, Any] = {
        "pre": options["pre"], "post": options["post"], "dictionary": release,
        "text_source": TEXT_SOURCE, "options": options, "sets": {},
    }
    for name, items in sequences(tokenizer, options, device).items():
        with step(f"{name}: {len(items)} sequences") as facts:
            diff = features.compare(pre, post, transcoders, items, name)
            fit = diff.fit(resamples=options["resamples"])
            report["sets"][name] = {
                "sequences": len(items), "positions": diff.positions,
                "fit": fit, "features": diff.feature_report(FLOOR),
            }
            facts["fvu"] = f"{fit['fvu_pre']['mean']:.3f} -> {fit['fvu_post']['mean']:.3f}"
            facts["visible_change"] = round(fit["visible_change"]["mean"], 3)
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, indent=1))
    for name, entry in report["sets"].items():
        fit = entry["fit"]
        low, high = fit["fvu_delta"]["mean_ci95"]
        moved = sum(layer["moved_across_halves"] for layer in entry["features"]["layers"])
        noise = sum(layer["moved_within_halves"] for layer in entry["features"]["layers"])
        log(f"{name:9s} fvu {fit['fvu_pre']['mean']:.3f} -> {fit['fvu_post']['mean']:.3f} "
            f"(delta 95% [{low:+.4f}, {high:+.4f}]), change visible {fit['visible_change']['mean']:.3f}, "
            f"features moved {moved:.0f} across halves vs {noise:.0f} within")
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
