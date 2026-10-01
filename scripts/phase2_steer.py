"""Phase 2: is SDFT's "example" direction a cause, and of what?

The activation-difference lens found that SDFT's mean residual-stream shift on
neutral prose promotes "example" in three languages, and the SDFT models' own
answers say why: they reason about "the example response" in 95 of 97 tool-use
answers and 465 of 507 science answers, an example only their teacher was ever
shown. This tests the direction behind it, two ways.

- **Teacher-forced**, on held-out text (wikitext validation and the science
  prompts with the base's answers): the probability that the next token is one
  of the "example" words. Base steered with the direction at 1x-8x its own norm
  (1x is what SDFT adds on average); SDFT with the direction projected out; a
  random direction of the same norm, both ways, as the control.
  The ablation that matters subtracts the mean difference (-1x): it undoes what
  SDFT added on average and nothing else. Projecting the direction out entirely
  is kept as a teacher-forced reference only -- it also removes whatever the
  base stream already had along it, and in a first pass it broke generation
  (0 of 8 correct, answers running to the length cap) while pushing the
  "example" probability below the base model's.
- **Generated**, on both eval sets and scored by the SDFT repo's own rule
  (`src/domains/lm/data/sdft.py::score`): SDFT as is, SDFT with the mean
  difference subtracted, and base steered. Accuracy, and how many answers mention an
  example. Greedy, Hugging Face generation rather than vLLM, so the unedited
  SDFT model is regenerated here as the reference rather than read from the
  vLLM eval.

If ablating the direction removes the confabulated example and leaves accuracy
alone, it is a by-product SDFT carries; if accuracy falls, the narration is part
of how the student does the task.

Run: uv run python -m scripts.phase2_steer
     uv run python -m scripts.phase2_steer --limit 32 --generate-limit 16      (a quick pass)
"""

import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from src.core.config import load_config
from src.domains.lm.analysis.actdiff import edited_stream
from src.domains.lm.data.sdft import TASKS, eval_items, score, task_texts
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-steer.json")
LOG = result("phase2-steer.log")
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()
MEANS = Path("results/actdiff-qwen3-0.6b/phase2-actdiff-means.safetensors")
BASE, SDFT = "qwen3-0.6b-sdft", "qwen3-0.6b-sdft-science"
SOURCE = f"{SDFT}|wikitext|early"
WORDS = [" example", " examples", " Example", "Example", "example", "例子", "示例", "のように", " Examples"]
MENTION = re.compile(r"\bexamples?\b|例子|示例|のように", re.IGNORECASE)
BUDGET = 2048


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def word_ids(tokenizer) -> List[int]:
    """The "example" words that are a single token in this vocabulary"""
    ids = []
    for word in WORDS:
        encoded = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    return ids


def example_logprob(model, sequences, ids: List[int]) -> float:
    """Mean over scored positions of log P(next token is an "example" word)"""
    import torch

    total, count = 0.0, 0
    with torch.no_grad():
        for tokens, start in sequences:
            logits = model(input_ids=tokens).logits[0, start - 1 : -1].float()
            mass = torch.logsumexp(logits.log_softmax(-1)[:, ids], dim=-1)
            total += mass.sum().item()
            count += mass.numel()
    return total / max(count, 1)


def generate(model, tokenizer, items: List[dict], batch: int, budget: int) -> List[str]:
    import torch

    tokenizer.padding_side = "left"
    texts = [tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
             for item in items]
    out = []
    for begin in range(0, len(texts), batch):
        encoded = tokenizer(texts[begin : begin + batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(model.device)
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        out.extend(tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True))
    return out


def judged(task: str, items: List[dict], responses: List[str]) -> Dict[str, Any]:
    marks = [score(task, response, item["answer"]) for response, item in zip(responses, items, strict=True)]
    return {
        "n": len(items),
        "accuracy": sum(marks) / len(marks),
        "correct": marks,
        "mention_example": sum(bool(MENTION.search(r)) for r in responses) / len(responses),
        "responses": responses,
    }


def parse(argv: List[str]) -> Dict[str, Any]:
    flags, index = {}, 0
    while index < len(argv):
        word = argv[index]
        if word.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else None
            if following is not None and not following.startswith("--"):
                flags[word[2:]] = following
                index += 1
            else:
                flags[word[2:]] = "1"
        index += 1
    return {
        "depth": float(flags.get("depth", 0.75)),
        "limit": int(flags.get("limit", 128)),
        "generate_limit": int(flags["generate-limit"]) if "generate-limit" in flags else None,
        "batch": int(flags.get("batch", 48)),
        "steer": float(flags.get("steer", 4.0)),
        # Hugging Face generation with hooks runs ~0.26 s a step at batch 48; the comparison is
        # paired (same items, with and without the edit), so a shorter budget costs both sides alike.
        "max_new_tokens": int(flags.get("max-new-tokens", BUDGET)),
        "seed": int(flags.get("seed", 0)),
    }


def main(argv: List[str]) -> int:
    import torch
    from datasets import load_dataset
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    options = parse(argv)
    set_log_file(LOG)
    guard(f"{BASE}..{SDFT}")
    tokenizer = AutoTokenizer.from_pretrained(load_config(BASE).hf_name)
    base, sdft = load(BASE), load(SDFT)
    device = base.device
    layer = dataclasses.replace(load_config(BASE), n_layers=base.config.num_hidden_layers).layer(options["depth"])
    direction = load_file(str(MEANS))[SOURCE][layer].to(device)
    generator = torch.Generator().manual_seed(options["seed"])
    random = torch.randn(direction.shape, generator=generator).to(device)
    random = random / random.norm() * direction.norm()
    ids = word_ids(tokenizer)
    banner("phase 2 steering the example direction", {
        **options, "layer": layer, "direction": SOURCE, "norm": round(float(direction.norm()), 3),
        "example words": [tokenizer.decode([i]) for i in ids], "artifact": ARTIFACT,
    })
    started = time.time()

    # Held-out text: wikitext validation (the direction came from the test split), and science prompts.
    prose = [row for row in load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")["text"]
             if len(row) > 1200][: options["limit"]]
    sets = {"wikitext-validation": [(tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
                                     ["input_ids"].to(device), 1) for t in prose]}
    sets["science-base-answers"] = []
    for item in task_texts(SDFT_REPO, "Qwen/Qwen3-0.6B", "science", tokenizer, BUDGET, options["limit"]):
        prompt = tokenizer(item.prompt, add_special_tokens=False)["input_ids"]
        answer = tokenizer(item.answer, add_special_tokens=False)["input_ids"]
        if len(answer) > 1:
            sets["science-base-answers"].append((torch.tensor([prompt + answer], device=device), len(prompt)))

    report: Dict[str, Any] = {"options": options, "layer": layer, "direction_norm": float(direction.norm()),
                              "example_ids": ids, "teacher_forced": {}, "generated": {}}
    conditions = [("base", base, {}), ("sdft", sdft, {}),
                  ("sdft - 1x direction", sdft, {"add": -direction}),
                  ("sdft - 1x random", sdft, {"add": -random}),
                  ("sdft, direction projected out", sdft, {"remove": direction}),
                  ("sdft, random projected out", sdft, {"remove": random})]
    conditions += [(f"base + {s:g}x direction", base, {"add": s * direction}) for s in (1, 2, 4, 8)]
    conditions += [("base + 8x random", base, {"add": 8 * random})]
    with step("teacher-forced log P(example word)"):
        for name, model, edit in conditions:
            with edited_stream(model, layer, **edit):
                report["teacher_forced"][name] = {set_name: example_logprob(model, seqs, ids)
                                                  for set_name, seqs in sets.items()}
            log(f"{name:24s} " + "  ".join(f"{k} {v:.3f}" for k, v in report["teacher_forced"][name].items()),
                indent=1)

    generated = [("sdft", sdft, {}), ("sdft - 1x direction", sdft, {"add": -direction}),
                 (f"base + {options['steer']:g}x direction", base, {"add": options["steer"] * direction})]
    for task in TASKS:
        items = eval_items(SDFT_REPO, task)[: options["generate_limit"]]
        report["generated"][task] = {}
        for name, model, edit in generated:
            with step(f"generating {task} ({len(items)}) under '{name}'") as facts, \
                    edited_stream(model, layer, **edit):
                verdict = judged(task, items, generate(model, tokenizer, items, options["batch"],
                                                        options["max_new_tokens"]))
                report["generated"][task][name] = verdict
                facts["accuracy"] = round(verdict["accuracy"], 3)
                facts["mention_example"] = round(verdict["mention_example"], 3)
            ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
            ARTIFACT.write_text(json.dumps(report, indent=1))
    for task, by_condition in report["generated"].items():
        for name, verdict in by_condition.items():
            log(f"{task:8s} {name:24s} accuracy {verdict['accuracy']:.3f}  "
                f"mentions example {verdict['mention_example']:.3f}")
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
