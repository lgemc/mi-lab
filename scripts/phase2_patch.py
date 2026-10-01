"""Phase 2: does the forgetting have an address? Patch one position's stream and see how much returns

The crosscoders found the most rotated shared features at the `</think>`
boundary, where reasoning ends and the answer format starts -- in SFT the same
latents fire before science's `<answer>` and tool use's `Action:`. If the
science stage's damage to tool use runs through that boundary, giving the
science model the tool-use model's residual stream at that one position should
bring back much of what was lost.

Same score as `phase2_restore`: teacher-forced on the before model's own correct
tool-use answers, mean log-probability of the answer tokens after the boundary.
The after model's stream after every block is overwritten, at the chosen
positions only, with the before model's on the same text:

- **`</think>`**: the boundary position;
- **one random answer position** (after the boundary), the control for "any one
  position";
- **every prompt position**: whether the forgetting is in how the tool
  documentation and question are read, rather than in how the answer is written.

Run: uv run python -m scripts.phase2_patch qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science
"""

import json
import random as stdlib_random
import sys
import time
from pathlib import Path
from typing import Dict, List

from src.core.config import load_config
from src.domains.lm.analysis.actdiff import capture_stream
from src.domains.lm.backend.layout import _blocks
from src.domains.lm.data.sdft import results_name
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-patch.json")
LOG = result("phase2-patch.log")
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
BOUNDARY = "</think>"


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def scored(model, ids, start: int) -> float:
    import torch

    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, start - 1 : -1].float()
        return float(logits.log_softmax(-1).gather(-1, ids[0, start:].unsqueeze(-1)).mean())


def patched_score(after, before_stream, positions: List[int], ids, start: int) -> float:
    """The after model's score with its stream after every block replaced at `positions` by the before model's"""
    index = list(positions)
    handles = []
    for layer, block in enumerate(_blocks(after)):
        def hook(_module, _inputs, out, at=layer):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden.clone()
            hidden[0, index] = before_stream[at, index].to(hidden.dtype)
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden
        handles.append(block.register_forward_hook(hook))
    try:
        return scored(after, ids, start)
    finally:
        for handle in handles:
            handle.remove()


def main(argv: List[str]) -> int:
    import torch
    from transformers import AutoTokenizer

    configs = [word for word in argv if not word.startswith("--")]
    if len(configs) != 2:
        raise SystemExit("give two configs: the model before the stage and after it")
    before_name, after_name = configs
    set_log_file(LOG)
    guard(f"{before_name}..{after_name}")
    tokenizer = AutoTokenizer.from_pretrained(load_config(before_name).hf_name)
    before, after = load(before_name), load(after_name)
    device = before.device
    boundary = tokenizer(BOUNDARY, add_special_tokens=False)["input_ids"]
    if len(boundary) != 1:
        raise SystemExit(f"'{BOUNDARY}' is {len(boundary)} tokens in this vocabulary; this script patches one position")
    boundary = boundary[0]
    saved = {name: json.loads((SDFT_REPO / "results" / results_name(load_config(name).hf_name, SDFT_REPO)
                               / "tooluse-2048" / "eval_responses.json").read_text())
             for name in (before_name, after_name)}
    banner("phase 2 position patching", {"before": before_name, "after": after_name, "artifact": ARTIFACT})
    started = time.time()
    generator = stdlib_random.Random(0)
    names = ("before", "after", "</think>", "random answer position", "every prompt position")
    rows: Dict[str, List[float]] = {name: [] for name in names}
    forgotten: List[bool] = []
    skipped = 0
    with step("patching"):
        for mine, theirs in zip(saved[before_name], saved[after_name], strict=True):
            if not mine["correct"]:
                continue
            prompt_ids = tokenizer(mine["prompt"], add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(mine["response"], add_special_tokens=False)["input_ids"]
            if boundary not in answer_ids:
                skipped += 1
                continue
            at = len(prompt_ids) + answer_ids.index(boundary)
            ids = torch.tensor([prompt_ids + answer_ids], device=device)
            start = at + 1
            if ids.shape[1] - start < 2:
                skipped += 1
                continue
            stream = capture_stream(before, ids)
            rows["before"].append(scored(before, ids, start))
            rows["after"].append(scored(after, ids, start))
            rows["</think>"].append(patched_score(after, stream, [at], ids, start))
            other = generator.randrange(start, ids.shape[1] - 1)
            rows["random answer position"].append(patched_score(after, stream, [other], ids, start))
            rows["every prompt position"].append(patched_score(after, stream, list(range(len(prompt_ids))), ids, start))
            forgotten.append(not theirs["correct"])

    def summary(keep: List[int]) -> Dict[str, object]:
        def fractions(pick):
            m = {name: sum(rows[name][i] for i in pick) / len(pick) for name in names}
            g = m["before"] - m["after"]
            return m, g, {name: (m[name] - m["after"]) / g if g else float("nan") for name in names[2:]}
        mean, gap, recovered = fractions(keep)
        draws = {name: [] for name in names[2:]}
        for _ in range(1000):
            _, _, sample = fractions([keep[generator.randrange(len(keep))] for _ in keep])
            for name in draws:
                draws[name].append(sample[name])
        return {"n": len(keep), "mean_logprob": mean, "gap": gap, "recovered": recovered,
                "recovered_ci95": {name: [sorted(v)[25], sorted(v)[974]] for name, v in draws.items()}}

    everything = list(range(len(rows["before"])))
    lost = [i for i in everything if forgotten[i]]
    report = {"before": before_name, "after": after_name, "skipped_no_boundary": skipped,
              "kept_by_before": summary(everything), "forgotten": summary(lost) if lost else None,
              "forgotten_flags": forgotten, "per_item": rows}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=1))
    for subset in ("kept_by_before", "forgotten"):
        entry = report[subset]
        if entry is None:
            continue
        log(f"{subset} (n={entry['n']}): gap {entry['gap']:.3f}")
        for name in names[2:]:
            low, high = entry["recovered_ci95"][name]
            log(f"  {name:24s} recovers {entry['recovered'][name]:+.3f}  [{low:+.3f}, {high:+.3f}]")
    log(f"skipped {skipped} answers with no '{BOUNDARY}'; done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
