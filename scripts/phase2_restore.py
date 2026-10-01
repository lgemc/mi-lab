"""Phase 2: which features carry the forgetting? Put them back and see how much returns

SFT's science stage cost ten tool-use questions; SDFT's cost two. The frozen
dictionary found SFT moved 4-5x as many tool-use-relevant features beyond noise.
This asks whether those features are the forgetting or merely beside it.

Teacher-forced on the *before* model's own correct behaviour: its saved greedy
answers (from `make eval`) on the tool-use items it got right, behind the
prompt it saw. The score is the mean log-probability of that answer. Before
(the tool-use model) minus after (the science model) is the forgetting gap on
this measure, and it is reported separately on the items that were actually
forgotten -- right before, wrong after. An intervention on the after model
recovers some fraction of the gap.

A first version scored a rendered golden tool call instead, and the "gap" came
out negative: the tool-use model was trained on demonstrations that open with a
thought line, so bare `Action:` lines were off-distribution for it in
particular. Scoring the model's own answers removes the format from the
question.

Interventions, all at the MLP outputs, the base dictionary's own site, with the
*before* model's values computed on the same text in a first pass:

- **top-shifted features**: in every layer, the 20 features whose firing rate
  moved most on tool-use text over this stage (`phase2-features.json`), their
  decoded contribution swapped from the after model's to the before model's;
- **random features**: as many features per layer, drawn from those active on
  this text in either model -- the control the top-shifted set has to beat;
- **all features**: the whole reconstruction swapped, the most the dictionary's
  visible seventh of the change can give back;
- **whole MLP output**: the after model's MLP output replaced by the before
  model's, error included -- the ceiling for any MLP-site restoration.

A fraction near the "all features" line for the top-shifted set, and far above
random, says the moved features are where the forgetting lives; near random
says the dictionary's view of the change misses it.

Run: uv run python -m scripts.phase2_restore qwen3-0.6b-sft-tooluse qwen3-0.6b-sft-science
"""

import json
import random as stdlib_random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import load_config
from src.domains.lm.analysis import features
from src.domains.lm.backend.layout import _blocks, _mlp
from src.domains.lm.data.sdft import results_name
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-restore.json")
LOG = result("phase2-restore.log")
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
DICTIONARY = "qwen3-0.6b-sdft"


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def answer_logprob(model, ids, start: int) -> float:
    import torch

    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, start - 1 : -1].float()
        chosen = logits.log_softmax(-1).gather(-1, ids[0, start:].unsqueeze(-1))
    return float(chosen.mean())


def patched(model, replace) -> List[Any]:
    """Hooks that let `replace(layer, input, output)` rewrite each block's MLP output

    The input is the one the MLP receives in *this* pass: with several layers
    patched at once, a later layer's input already carries the earlier patches,
    and encoding the unpatched run's input there would swap the wrong features.
    """
    handles = []
    for index, block in enumerate(_blocks(model)):
        def hook(_module, inputs, out, at=index):
            return replace(at, inputs[0][0], out)
        handles.append(_mlp(block, index).register_forward_hook(hook))
    return handles


def swap(transcoders, before_inputs, chosen: Optional[Dict[int, Any]], whole: bool):
    """A replace() that swaps the chosen features' contribution (or everything) for the before model's"""

    def replace(layer, x_after, out):
        x_before = before_inputs[layer]
        if whole:
            return before_outputs_cache[layer].to(out.dtype).unsqueeze(0)
        transcoder = transcoders[layer]
        f_after, f_before = transcoder.encode(x_after), transcoder.encode(x_before)
        if chosen is not None:
            mask = chosen[layer]
            f_after, f_before = f_after[:, mask], f_before[:, mask]
            decoder = transcoder.W_dec[mask]
        else:
            decoder = transcoder.W_dec
        delta = (f_before - f_after) @ decoder
        return out + delta.to(out.dtype).unsqueeze(0)

    return replace


# The before model's MLP outputs for the current sequence, read by the whole-output swap.
before_outputs_cache: Dict[int, Any] = {}


def main(argv: List[str]) -> int:
    import torch
    from transformers import AutoTokenizer

    configs = [word for word in argv if not word.startswith("--")]
    if len(configs) != 2:
        raise SystemExit("give two configs: the model before the stage and after it")
    before_name, after_name = configs
    set_log_file(LOG)
    guard(f"{before_name}..{after_name}")
    pair_dir = Path("results") / {
        ("qwen3-0.6b-sft-tooluse", "qwen3-0.6b-sft-science"): "diff-qwen3-0.6b-sft-tooluse-to-science",
        ("qwen3-0.6b-sdft-tooluse", "qwen3-0.6b-sdft-science"): "diff-qwen3-0.6b-sdft-tooluse-to-science",
    }[(before_name, after_name)]
    shifted = json.loads((pair_dir / "phase2-features.json").read_text())["sets"]["tooluse"]["features"]["layers"]
    release = load_config(DICTIONARY).transcoder.release
    tokenizer = AutoTokenizer.from_pretrained(load_config(before_name).hf_name)
    before, after = load(before_name), load(after_name)
    device = before.device
    transcoders = features.load_transcoders(release, before.config.num_hidden_layers, str(device))
    width = transcoders[0].W_enc.shape[0]
    banner("phase 2 restoring features", {"before": before_name, "after": after_name, "dictionary": release,
                                          "shifted from": str(pair_dir), "artifact": ARTIFACT})
    started = time.time()

    saved = {name: json.loads((SDFT_REPO / "results" / results_name(load_config(name).hf_name, SDFT_REPO)
                               / "tooluse-2048" / "eval_responses.json").read_text())
             for name in (before_name, after_name)}
    sequences, forgotten = [], []
    for mine, theirs in zip(saved[before_name], saved[after_name], strict=True):
        if not mine["correct"]:
            continue
        prompt_ids = tokenizer(mine["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(mine["response"], add_special_tokens=False)["input_ids"]
        sequences.append((torch.tensor([prompt_ids + answer_ids], device=device), len(prompt_ids)))
        forgotten.append(not theirs["correct"])

    top = {entry["layer"]: torch.tensor([f["feature"] for f in entry["top_shifted"]], device=device)
           for entry in shifted}
    count = {layer: len(index) for layer, index in top.items()}
    rows: Dict[str, List[float]] = {name: [] for name in
                                    ("before", "after", "top-shifted", "random", "all features", "whole MLP output")}
    generator = stdlib_random.Random(0)
    with step(f"{len(sequences)} sequences, six conditions each"):
        for ids, start in sequences:
            in_before, out_before = features.capture_mlp(before, ids)
            in_after, _ = features.capture_mlp(after, ids)
            before_inputs = {layer: in_before[layer] for layer in range(in_before.shape[0])}
            before_outputs_cache.clear()
            before_outputs_cache.update({layer: out_before[layer] for layer in range(out_before.shape[0])})
            # The random control draws, per layer, among features active on this sequence in either model.
            chosen_random = {}
            for layer, transcoder in enumerate(transcoders):
                with torch.no_grad():
                    active = ((transcoder.encode(in_before[layer]) > 0).any(0)
                              | (transcoder.encode(in_after[layer]) > 0).any(0)).nonzero().flatten().tolist()
                pool = active if len(active) >= count[layer] else list(range(width))
                chosen_random[layer] = torch.tensor(generator.sample(pool, count[layer]), device=device)

            rows["before"].append(answer_logprob(before, ids, start))
            rows["after"].append(answer_logprob(after, ids, start))
            for name, chosen, whole in (("top-shifted", top, False), ("random", chosen_random, False),
                                        ("all features", None, False), ("whole MLP output", None, True)):
                handles = patched(after, swap(transcoders, before_inputs, chosen, whole))
                try:
                    rows[name].append(answer_logprob(after, ids, start))
                finally:
                    for handle in handles:
                        handle.remove()

    conditions = ("top-shifted", "random", "all features", "whole MLP output")

    def summary(keep: List[int]) -> Dict[str, Any]:
        """Mean log-probabilities and recovered fractions over a subset, with a paired bootstrap"""
        def fractions(pick):
            m = {name: sum(rows[name][i] for i in pick) / len(pick) for name in rows}
            g = m["before"] - m["after"]
            return m, g, {name: (m[name] - m["after"]) / g if g else float("nan") for name in conditions}
        mean, gap, recovered = fractions(keep)
        draws = {name: [] for name in conditions}
        for _ in range(1000):
            _, _, sample = fractions([keep[generator.randrange(len(keep))] for _ in keep])
            for name in conditions:
                draws[name].append(sample[name])
        intervals = {name: [sorted(v)[25], sorted(v)[974]] for name, v in draws.items()}
        return {"n": len(keep), "mean_logprob": mean, "gap": gap, "recovered": recovered,
                "recovered_ci95": intervals}

    everything = list(range(len(sequences)))
    lost = [i for i in everything if forgotten[i]]
    report = {"before": before_name, "after": after_name, "dictionary": release,
              "features_per_layer": count, "kept_by_before": summary(everything),
              "forgotten": summary(lost) if lost else None, "forgotten_flags": forgotten, "per_item": rows}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(report, indent=1))
    for subset in ("kept_by_before", "forgotten"):
        entry = report[subset]
        if entry is None:
            continue
        mean = entry["mean_logprob"]
        log(f"{subset} (n={entry['n']}): log P before {mean['before']:.3f}, after {mean['after']:.3f}, "
            f"gap {entry['gap']:.3f}")
        for name in conditions:
            low, high = entry["recovered_ci95"][name]
            log(f"  {name:18s} recovers {entry['recovered'][name]:+.3f}  [{low:+.3f}, {high:+.3f}]")
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
