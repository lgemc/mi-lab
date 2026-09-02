"""Load a learned circuit into the model and run it, on data it has never seen.

Every number a sheaf reports comes from the same objective it was trained on:
a two-way choice between the answer token and one distractor, scored as a logit
comparison. That is a *ranking* test, and a mask can pass it with a route that
could not produce the answer if asked. On GPT-2 small the mask found exactly
such a route -- 0.969 held out while the model's own name-mover heads could be
deleted from it with no effect at all -- so the question this script exists for
is whether the circuit is a circuit or a scoring trick.

Three things it does that the training loop cannot:

  fresh data      a new draw of the task at a different seed. The holdout the
                  run reported was held out of *training*, but it came from the
                  same pool, split once. This builds the task again from
                  scratch, so nothing here was chosen when the mask was.
  generation      the model is asked to *produce* the continuation rather than
                  rank two candidates. A shortcut that ranks well can still
                  generate nothing usable, and that gap is the finding.
  a real load     the mask is multiplied into the weights and the model runs
                  normally -- no functional_call, no gradient, no gates in the
                  forward pass. If a circuit cannot be loaded and run like a
                  model, it was never a runnable object.

Masked weights are restored on the way out, so the adapter is reusable and a
mistake here cannot quietly poison a later measurement in the same process.

A common pipe could be: gates | apply | fresh task | rank + generate | compare

Run: uv run python -m scripts.sheaf_infer gpt2-small results/gpt2-sweep/s0.1-seeded
     uv run python -m scripts.sheaf_infer gpt2-small <dir> --task ioi --seed 99 --show 8
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from scripts.observe import banner, log
from src.data.tasks import build_task
from src.methods.circuits import require_circuits
from src.methods.sheaves import gateable
from src.model.adapter import load_adapter


@contextmanager
def circuit_loaded(adapter, gates: dict):
    """Multiply the thresholded mask into the weights, and put them back after

    In place, because inference needs no gradient and `functional_call` exists
    in the training loop only to keep one. This is what "a circuit you can run"
    has to mean: ordinary weights, ordinary forward pass.
    """
    targets = gateable(adapter)
    saved = {}
    try:
        with torch.no_grad():
            for name, parameter in targets.items():
                if name not in gates:
                    continue
                saved[name] = parameter.detach().clone()
                parameter.mul_((gates[name].to(parameter.device) > 0).to(parameter.dtype))
        yield
    finally:
        with torch.no_grad():
            for name, original in saved.items():
                targets[name].copy_(original)

def ranking(adapter, task) -> float:
    """The training objective's own metric: does the answer outrank the distractor"""
    prompts = list(task.clean)
    io, subject = task.answers(adapter)
    logits = adapter.logits(prompts)
    right = sum(1 for row in range(len(prompts))
                if float(logits[row, io[row]]) > float(logits[row, subject[row]]))
    return right / len(prompts)

def generation(adapter, task, tokens: int) -> dict:
    """Ask for the continuation and see whether the answer is what comes out

    `first_token` is the fair comparison with the ranking score -- both are
    about the very next token -- while `contains` allows the answer to arrive
    a word or two late, which greedy decoding on a templated frame often does.
    """
    prompts = list(task.clean)
    # Decoded from the ids the CircuitTask protocol returns, not read off the
    # examples: IOIExample carries `io`/`subject` as bare names while
    # TemplateTask's carries `answer`/`distractor` with the leading space, and
    # `answers()` is the one shape every task agrees on.
    io_ids, subject_ids = task.answers(adapter)
    answers = [adapter.tokenizer.decode([i]) for i in io_ids]
    distractors = [adapter.tokenizer.decode([i]) for i in subject_ids]
    completions = adapter.generate(prompts, max_new_tokens=tokens)
    exact = sum(1 for c, a in zip(completions, answers, strict=True)
                if c.startswith(a))
    contains = sum(1 for c, a in zip(completions, answers, strict=True)
                   if a.strip() and a.strip() in c)
    wrong = sum(1 for c, d in zip(completions, distractors, strict=True)
                if d.strip() and d.strip() in c)
    return {
        "first_token": exact / len(prompts),
        "contains_answer": contains / len(prompts),
        "contains_distractor": wrong / len(prompts),
        "completions": completions,
        "answers": answers,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("directory", help="results dir holding sheaf-<task>-gates.pt")
    parser.add_argument("--task", default="ioi")
    parser.add_argument("--size", type=int, default=128, help="fresh prompts to draw")
    parser.add_argument("--seed", type=int, default=99,
                        help="NOT the training seed; this draws a task the mask never saw")
    parser.add_argument("--tokens", type=int, default=6)
    parser.add_argument("--show", type=int, default=6, help="example completions to print")
    args = parser.parse_args()

    directory = Path(args.directory)
    gates_path = directory / f"sheaf-{args.task}-gates.pt"
    if not gates_path.exists():
        raise SystemExit(f"{gates_path} does not exist; rerun that point with gates saved")
    gates = torch.load(gates_path, weights_only=True)
    adapter = require_circuits(load_adapter(args.config))
    task = build_task(args.task, adapter, size=args.size, seed=args.seed)
    total = sum(int(v.numel()) for v in gates.values())
    opened = sum(int((v > 0).sum()) for v in gates.values())

    banner("circuit inference", {
        "config": args.config,
        "circuit": f"{opened} of {total} weights ({opened / total:.4%})",
        "task": f"{args.task}, {len(set(task.clean))} distinct of {args.size} fresh prompts",
        "seed": f"{args.seed} (the run trained at its own seed; this draw is new)",
    })

    full_rank = ranking(adapter, task)
    full_gen = generation(adapter, task, args.tokens)
    with circuit_loaded(adapter, gates):
        circuit_rank = ranking(adapter, task)
        circuit_gen = generation(adapter, task, args.tokens)

    log("")
    log(f"{'':<22} {'full model':>12} {'circuit':>12} {'delta':>9}")
    for label, a, b in (
        ("ranking (2AFC)", full_rank, circuit_rank),
        ("generates answer 1st", full_gen["first_token"], circuit_gen["first_token"]),
        ("answer anywhere", full_gen["contains_answer"], circuit_gen["contains_answer"]),
        ("distractor anywhere", full_gen["contains_distractor"], circuit_gen["contains_distractor"]),
    ):
        log(f"{label:<22} {a:>12.3f} {b:>12.3f} {b - a:>+9.3f}")
    log("")
    log("chance on the ranking test is 0.500; on generation it is ~0")

    log("")
    log("sample completions (prompt -> full model | circuit):")
    for row in range(min(args.show, len(task.examples))):
        log(f"  {list(task.clean)[row][-58:]!r}")
        log(f"      want {full_gen['answers'][row]!r}   "
            f"full {full_gen['completions'][row][:26]!r}   "
            f"circuit {circuit_gen['completions'][row][:26]!r}")

    out = directory / f"sheaf-{args.task}-inference.json"
    out.write_text(json.dumps({
        "protocol": "the circuit multiplied into the weights and run as an ordinary model, on a "
                    "fresh draw of the task at a seed the run never used. Ranking is the objective "
                    "the mask was trained on; generation is the question it was never asked, and "
                    "the gap between them is the point.",
        "config": args.config, "task": args.task, "seed": args.seed, "size": args.size,
        "n_gates": total, "n_open": opened, "density": round(opened / total, 8),
        "full_model": {"ranking": round(full_rank, 4),
                       **{k: round(v, 4) for k, v in full_gen.items()
                          if k not in ("completions", "answers")}},
        "circuit": {"ranking": round(circuit_rank, 4),
                    **{k: round(v, 4) for k, v in circuit_gen.items()
                       if k not in ("completions", "answers")}},
        "samples": [
            {"prompt": list(task.clean)[r], "answer": full_gen["answers"][r],
             "full": full_gen["completions"][r], "circuit": circuit_gen["completions"][r]}
            for r in range(min(args.show, len(task.examples)))
        ],
        "command": f"uv run python -m scripts.sheaf_infer {args.config} {directory} "
                   f"--task {args.task} --seed {args.seed}",
    }, indent=2) + "\n")
    log(f"-> {out}")

if __name__ == "__main__":
    main()
