"""Phase 3: is the forgetting in a few directions of the update, or in its tail?

Proposal 04 (`~/main/m/proposals/sdft-model-diffing/`) asks whether parameter
decomposition can diff a fine-tune. Before decomposing anything, the plain-SVD
question: write the science stage's update to every block linear map,
D = W_science - W_tooluse, in its singular basis, and build models that keep
only part of it --

- **top k**: W_before + the k leading directions of each D;
- **all but top k**: W_before + everything except them;
- **linears, all of D**: every block map at the after model's weights, the
  embeddings and norms still at the before model's (their share is the gap
  between this row and `after`).

Each is scored by `stage`: how much of the forgetting it keeps, how
much of the learning, how far it drifts from `after` on wikitext. If a small k
keeps the learning but not the forgetting, the forgetting lives in the tail
and a rank-k projection of the update would have avoided it; if the two move
together, the plain basis cannot separate them, and that is the job the
decomposition (`phase3_delta_pd`) is for.

Also written: each map's spectrum (stable rank, rank holding 50/90/99% of the
energy, the leading singular values).

Run: uv run python -m scripts.phase3_delta_rank qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science
     ... --export=8    (write that truncation as a checkpoint for `make eval`, and score nothing)
"""

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from src.core.config import load_config
from src.domains.lm.analysis import delta, stage
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

RANKS = [1, 8, 32, 128, 512]
# The generation budget the saved eval answers were made under; part of their directory name.
BUDGET = 2048


def export(work, tokenizer, base, factors, after_name: str, top: int) -> int:
    """Write the before model plus all but the top `top` directions of the update, for `make eval`

    Into the Self-Distillation repo's runs/derived/, so the eval names its
    results directory the way it names every other checkpoint's.
    """
    source = Path(load_config(after_name).hf_name)
    target = stage.SDFT_REPO / "runs" / "derived" / f"{source.parent.name}-{source.name}-allbut{top}"
    with step(f"exporting all but the top {top} to {target}"):
        delta.truncate(work, base, factors, top, complement=True)
        work.save_pretrained(target)
        tokenizer.save_pretrained(target)
    log(f"evaluate with: make eval DATASET=tooluse CKPT={target.relative_to(stage.SDFT_REPO)} "
        f"(and DATASET=science) in {stage.SDFT_REPO}")
    return 0


def main(argv: List[str]) -> int:
    import torch
    from transformers import AutoTokenizer

    configs = [word for word in argv if not word.startswith("--")]
    flags = dict(word[2:].split("=", 1) for word in argv if word.startswith("--") and "=" in word)
    if len(configs) != 2:
        raise SystemExit("give two configs: the model before the stage and after it")
    before_name, after_name = configs
    ranks = [int(r) for r in flags.get("ranks", ",".join(map(str, RANKS))).split(",")]
    artifact, log_path = result("phase3-delta-rank.json"), result("phase3-delta-rank.log")
    set_log_file(log_path)
    guard(f"{before_name}..{after_name}")
    banner("phase 3 truncating the update", {"before": before_name, "after": after_name,
                                             "ranks": ranks, "artifact": artifact})
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(load_config(before_name).hf_name)
    before, after = stage.load(before_name), stage.load(after_name)
    device = next(before.parameters()).device

    with step("updates and their spectra"):
        deltas = delta.updates(before, after, device=str(device))
        base = {name: module.weight.detach().float().clone() for name, module in delta.targets(before).items()}
        spectra = [delta.Spectrum.of(name, deltas[name], base[name]) for name in deltas]
        factors = delta.Factors(deltas, str(device))
    if "export" in flags:
        return export(before, tokenizer, base, factors, after_name, int(flags["export"]))
    with step("scoring sets"):
        sets = stage.build(before_name, after_name, tokenizer, device, BUDGET,
                            later_limit=int(flags.get("later", 80)), prose=int(flags.get("prose", 16)))
        ref = stage.reference(after, sets)
    log(f"sets: {sets.meta}")

    rows: Dict[str, Dict[str, List[float]]] = {}
    with step("before and after"):
        rows["before"] = stage.score(before, sets, ref)
        rows["after"] = stage.score(after, sets, ref)
    del after
    work = before  # the before model becomes the working copy; `base` keeps its weights
    variants = [("linears, all of D", None, False)]
    variants += [(f"top {k}", k, False) for k in ranks]
    variants += [(f"all but top {k}", k, True) for k in ranks]
    for name, top, complement in variants:
        with step(name):
            delta.truncate(work, base, factors, top, complement)
            rows[name] = stage.score(work, sets, ref)
            entry = stage.fractions(rows, [name], sets, draws=200)[name]
            log(stage.line(name, entry))
        torch.cuda.empty_cache()

    names = [name for name, _, _ in variants]
    report: Dict[str, Any] = {
        "before": before_name, "after": after_name, "sets": sets.meta, "ranks": ranks,
        "variants": stage.fractions(rows, names, sets),
        "spectra": [asdict(s) for s in spectra],
        "per_item": rows,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=1))
    log("--- final, 1000-draw intervals ---")
    for name in names:
        log(stage.line(name, report["variants"][name]))
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MI_LAB_RESULTS", stage.pair_dir(*[w for w in sys.argv[1:] if not w.startswith("--")][:2]))
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
