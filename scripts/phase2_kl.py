"""Phase 2: how far each fine-tune's output distribution moved from the base model's

The second cheap diagnostic of `~/main/m/proposals/model-diffing-sdft-transcoders.md`
(experiment 4, last row): RL's Razor (2509.04259) finds that forgetting tracks the
KL divergence between the fine-tuned policy and the base policy, measured on the
new task. `phase2_diffing weights` says how far the *weights* moved; this says how
far the *behaviour* did, which is the quantity the forgetting claim is about.

Per position, the full-vocabulary KL between the two next-token distributions,
in both directions, averaged over tokens:

- **On each task's eval prompts, along the fine-tuned model's own answers.** These
  are the greedy responses `make eval` saved in `~/main/Self-Distillation/results`,
  so no generation happens here. The answer is the fine-tuned model's, which
  makes this the on-policy KL RL's Razor measures, with greedy decoding standing
  in for sampling. Run over both tasks, so a science-stage model also reports how
  far it moved on tool-use prompts: the prior task, where forgetting would show.
- **On neutral prose** (wikitext), teacher-forced: drift on text neither
  fine-tune saw.

Run: uv run python -m scripts.phase2_kl
     uv run python -m scripts.phase2_kl qwen3-0.6b-sdft qwen3-0.6b-sft-tooluse --passages 16 --limit 50
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.config import load_config
from src.domains.lm.data.wikitext import passages
from src.telemetry.observe import banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACT = result("phase2-kl.json")
LOG = result("phase2-kl.log")

POSTS = [
    "qwen3-0.6b-sdft-tooluse",
    "qwen3-0.6b-sft-tooluse",
    "qwen3-0.6b-sdft-science",
    "qwen3-0.6b-sft-science",
]
TASKS = ["tooluse", "science"]
SDFT_REPO = Path("~/main/Self-Distillation").expanduser()

# Positions per slice of the full-vocabulary computation: a 3k-token answer's
# logits are ~2 GB per model in float32, so the KL is taken a slice at a time.
CHUNK = 512
# Wikitext passages are cut to this many tokens; long enough to be prose, short
# enough that the passage set costs seconds.
PASSAGE_TOKENS = 1024


class KLRunError(RuntimeError):
    """A phase 2 KL run that cannot proceed, said with the argument that would fix it"""


def responses(config: str, task: str, budget: int) -> List[Dict[str, Any]]:
    """The fine-tuned model's own saved eval answers for one task

    Found the way the Self-Distillation Makefile names them: the checkpoint path
    relative to that repo, slashes turned into underscores.
    """
    checkpoint = Path(load_config(config).hf_name)
    try:
        name = str(checkpoint.relative_to(SDFT_REPO)).replace("/", "_")
    except ValueError as error:
        raise KLRunError(
            f"{config} points at {checkpoint}, outside {SDFT_REPO}, so there are no saved eval answers to read"
        ) from error
    path = SDFT_REPO / "results" / name / f"{task}-{budget}" / "eval_responses.json"
    if not path.exists():
        raise KLRunError(f"{path} does not exist; run `make eval DATASET={task} CKPT=...` in {SDFT_REPO} first")
    return json.loads(path.read_text())


def load(config: str):
    import torch
    from transformers import AutoModelForCausalLM

    cfg = load_config(config)
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=getattr(torch, cfg.dtype))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def kl_both_ways(base, post, ids, start: int) -> Tuple[float, float, int]:
    """Summed KL(post||base) and KL(base||post) over positions predicting ids[start:]

    Position t's distribution predicts token t+1, so the positions read are
    start-1 .. len-2. Returned as sums with a token count, so callers can weight
    by tokens across sequences of different length.
    """
    import torch

    with torch.no_grad():
        logits_base = base(input_ids=ids).logits[0, start - 1 : -1]
        logits_post = post(input_ids=ids).logits[0, start - 1 : -1]
        forward = backward = 0.0
        for at in range(0, logits_base.shape[0], CHUNK):
            lb = logits_base[at : at + CHUNK].float().log_softmax(-1)
            lp = logits_post[at : at + CHUNK].float().log_softmax(-1)
            forward += (lp.exp() * (lp - lb)).sum().item()
            backward += (lb.exp() * (lb - lp)).sum().item()
    return forward, backward, logits_base.shape[0]


def summarise(rows: List[Tuple[float, float, int]]) -> Dict[str, float]:
    tokens = sum(n for _, _, n in rows)
    if not tokens:
        return {"sequences": len(rows), "tokens": 0}
    return {
        "sequences": len(rows),
        "tokens": tokens,
        "kl_post_base": sum(f for f, _, _ in rows) / tokens,
        "kl_base_post": sum(b for _, b, _ in rows) / tokens,
    }


def measure(pre: str, post: str, base, tokenizer, texts: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
    """Every set for one fine-tuned checkpoint, against the resident base model"""
    import torch

    model = load(post)
    device = next(model.parameters()).device
    report: Dict[str, Any] = {}
    for task in TASKS:
        items = responses(post, task, options["budget"])[: options["limit"]]
        with step(f"{post} on its own {task} answers ({len(items)})") as facts:
            rows = []
            for item in items:
                text = item["prompt"]
                if not isinstance(text, str):
                    # The science eval saves messages and templates them at
                    # generation time; this repeats that call.
                    text = tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
                prompt = tokenizer(text, add_special_tokens=False)["input_ids"]
                answer = tokenizer(item["response"], add_special_tokens=False)["input_ids"]
                if not answer:
                    continue
                ids = torch.tensor([prompt + answer], device=device)
                rows.append(kl_both_ways(base, model, ids, len(prompt)))
            report[task] = summarise(rows)
            facts["kl_post_base"] = round(report[task].get("kl_post_base", float("nan")), 4)
    with step(f"{post} on {len(texts)} wikitext passages") as facts:
        rows = []
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=PASSAGE_TOKENS)
            ids = encoded["input_ids"].to(device)
            rows.append(kl_both_ways(base, model, ids, 1))
        report["wikitext"] = summarise(rows)
        facts["kl_post_base"] = round(report["wikitext"]["kl_post_base"], 4)
        facts["gpu"] = gpu()
    del model
    torch.cuda.empty_cache()
    return report


def parse(argv: List[str]) -> Dict[str, Any]:
    # A flag's value is not a config name: `--limit 3` must not add a model called "3".
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
    return {
        "pre": positional[0] if positional else "qwen3-0.6b-sdft",
        "posts": positional[1:] or POSTS,
        "passages": int(flags.get("passages", 32)),
        "limit": int(flags["limit"]) if "limit" in flags else None,
        "budget": int(flags.get("budget", 2048)),
    }


def main(argv: List[str]) -> int:
    from transformers import AutoTokenizer

    options = parse(argv)
    set_log_file(LOG)
    guard(f"{options['pre']}..{'+'.join(options['posts'])}")
    banner("phase 2 output KL against the base model", {
        "pre": options["pre"],
        "posts": ", ".join(options["posts"]),
        "tasks": ", ".join(TASKS),
        "passages": options["passages"],
        "limit": options["limit"] or "all",
        "artifact": ARTIFACT,
    })
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(load_config(options["pre"]).hf_name)
    texts = passages(options["passages"])
    with step(f"loading {options['pre']}"):
        base = load(options["pre"])
    report = {"pre": options["pre"], "options": {k: v for k, v in options.items() if k != "posts"}, "models": {}}
    for post in options["posts"]:
        report["models"][post] = measure(options["pre"], post, base, tokenizer, texts, options)
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, indent=2))
    for post, sets in report["models"].items():
        measured = [f"{name} {s['kl_post_base']:.4f}" for name, s in sets.items() if s.get("tokens")]
        log(f"{post:26s} " + "  ".join(measured))
    log(f"done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-diffing", outputs=[results_root()]))
