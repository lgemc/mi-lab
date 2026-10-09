"""What a model between two stage checkpoints is scored on: how much it forgot, learned, drifted

A stage takes a model from `before` (it knows the earlier task) to `after` (it
learned the later one and, for SFT, forgot some of the earlier). Each phase 3
tool builds models between the two and asks three things of each:

- **forgetting kept**, on the earlier task: the mean log-probability of the
  *before* model's own correct answers (its saved greedy outputs, behind the
  prompt it saw), as in `phase2_restore`. 1 means the variant lost as much as
  `after` did, 0 means it lost none. Reported on every kept item and on the
  items `after` actually got wrong.
- **learning kept**, on the later task: the same score on the *after* model's
  correct answers. 1 means the variant gained as much as `after`, 0 none.
- **drift** on wikitext: KL(after || variant), teacher-forced.

Every fraction comes with a paired bootstrap interval over items.

The saved answers are the ones `make eval` wrote in the Self-Distillation repo
under a generation budget the caller names, since that number is part of the
results directory's name.

A common pipe could be: build | reference | score (before, after, variants) | fractions
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....core.config import load_config
from ..data.sdft import results_name
from ..data.wikitext import passages

SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
EARLIER, LATER = "tooluse", "science"


def load(config: str, device: Optional[str] = None):
    import torch
    from transformers import AutoModelForCausalLM

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to(device).eval()


def saved(config: str, task: str, budget: int) -> List[Dict[str, Any]]:
    name = results_name(load_config(config).hf_name, SDFT_REPO)
    return json.loads((SDFT_REPO / "results" / name / f"{task}-{budget}" / "eval_responses.json").read_text())


@dataclass
class Item:
    ids: Any
    start: int
    forgotten: bool = False


@dataclass
class Sets:
    earlier: List[Item]
    later: List[Item]
    prose: List[Any]
    meta: Dict[str, Any] = field(default_factory=dict)


def item(tokenizer, prompt, answer: str, device) -> Item:
    import torch

    if not isinstance(prompt, str):
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    head = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    tail = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return Item(ids=torch.tensor([head + tail], device=device), start=len(head))


def build(before: str, after: str, tokenizer, device, budget: int, later_limit: int, prose: int,
          prose_tokens: int = 512, prose_skip: int = 0) -> Sets:
    """The scoring sets for one stage

    `prose_skip` drops the first passages, so a tool that trains on wikitext can
    score on passages it did not train on.
    """
    earlier = []
    for mine, theirs in zip(saved(before, EARLIER, budget), saved(after, EARLIER, budget), strict=True):
        if mine["correct"] and mine["response"]:
            entry = item(tokenizer, mine["prompt"], mine["response"], device)
            entry.forgotten = not theirs["correct"]
            earlier.append(entry)
    later = [item(tokenizer, row["prompt"], row["response"], device)
             for row in saved(after, LATER, budget) if row["correct"] and row["response"]][:later_limit]
    texts = passages(prose + prose_skip)[prose_skip:]
    encoded = [tokenizer(text, return_tensors="pt", truncation=True, max_length=prose_tokens)["input_ids"].to(device)
               for text in texts]
    return Sets(earlier=earlier, later=later, prose=encoded,
                meta={"earlier": len(earlier), "forgotten": sum(i.forgotten for i in earlier),
                      "later": len(later), "prose": len(encoded)})


def answer_logprob(model, entry: Item) -> float:
    import torch

    with torch.no_grad():
        logits = model(input_ids=entry.ids).logits[0, entry.start - 1: -1].float()
        chosen = logits.log_softmax(-1).gather(-1, entry.ids[0, entry.start:].unsqueeze(-1))
    return float(chosen.mean())


def prose_kl(model, reference_logits: List[Any]) -> List[float]:
    import torch

    out = []
    with torch.no_grad():
        for ids, ref in reference_logits:
            lp = model(input_ids=ids).logits[0].float().log_softmax(-1)
            out.append(float((ref.exp() * (ref - lp)).sum(-1).mean()))
    return out


def reference(model, sets: Sets) -> List[Any]:
    """The after model's log-distributions on the prose, kept on the device for the drift"""
    import torch

    with torch.no_grad():
        return [(ids, model(input_ids=ids).logits[0].float().log_softmax(-1)) for ids in sets.prose]


def score(model, sets: Sets, ref) -> Dict[str, List[float]]:
    return {"earlier": [answer_logprob(model, e) for e in sets.earlier],
            "later": [answer_logprob(model, e) for e in sets.later],
            "drift": prose_kl(model, ref)}


def fractions(rows: Dict[str, Dict[str, List[float]]], names: List[str], sets: Sets,
              seed: int = 0, draws: int = 1000) -> Dict[str, Dict[str, Any]]:
    """Forgetting kept and learning kept per variant, with paired 95% bootstrap intervals

    `rows` must hold "before" and "after" beside the variants.
    """
    generator = random.Random(seed)
    lost = [i for i, e in enumerate(sets.earlier) if e.forgotten]

    def kept(series: str, pick: List[int], name: str) -> float:
        def mean(key: str) -> float:
            return sum(rows[key][series][i] for i in pick) / len(pick)

        gap = mean("after") - mean("before")
        return (mean(name) - mean("before")) / gap if gap else float("nan")

    def interval(series: str, pool: List[int], name: str) -> List[float]:
        values = sorted(kept(series, [pool[generator.randrange(len(pool))] for _ in pool], name)
                        for _ in range(draws))
        return [values[int(0.025 * draws)], values[int(0.975 * draws) - 1]]

    everything = list(range(len(sets.earlier)))
    later = list(range(len(sets.later)))
    out = {}
    for name in names:
        entry = {
            "forgetting_kept": kept("earlier", everything, name),
            "forgetting_kept_ci95": interval("earlier", everything, name),
            "learning_kept": kept("later", later, name),
            "learning_kept_ci95": interval("later", later, name),
            "drift_kl_after": sum(rows[name]["drift"]) / len(rows[name]["drift"]),
        }
        if lost:
            entry["forgetting_kept_on_forgotten"] = kept("earlier", lost, name)
            entry["forgetting_kept_on_forgotten_ci95"] = interval("earlier", lost, name)
        out[name] = entry
    return out


def line(name: str, entry: Dict[str, Any]) -> str:
    lost, (lost_low, lost_high) = entry["forgetting_kept"], entry["forgetting_kept_ci95"]
    gained, (gained_low, gained_high) = entry["learning_kept"], entry["learning_kept_ci95"]
    return (f"{name:28s} forgetting kept {lost:+.3f} [{lost_low:+.3f},{lost_high:+.3f}]   "
            f"learning kept {gained:+.3f} [{gained_low:+.3f},{gained_high:+.3f}]   "
            f"drift {entry['drift_kl_after']:.4f}")


def pair_dir(before: str, after: str) -> str:
    """results/diff-<model>-<method>-tooluse-to-science[-s43], the directory phase 2 used for this stage"""
    method = "sft" if "-sft-" in before else "sdft"
    seed = "-s43" if before.endswith("-s43") else ""
    return f"results/diff-qwen3-0.6b-{method}-tooluse-to-science{seed}"

