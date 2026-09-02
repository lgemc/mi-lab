"""Run DiscoGP weight pruning on a task, which until now had no way to be run.

`methods/sheaves.py` has been exercised only from the test suite and a REPL, on
GPT-2 small and on IOI, and both of those were accidents of what was cheap
rather than choices. `prune` takes any `CircuitTask`, so translation was always
one argument away; this is that argument, plus the bookkeeping that makes a run
an artifact instead of a number in a terminal that is now closed.

Two things it refuses to do quietly. It will not start a run whose optimizer
state does not fit in memory -- four float32 tensors per gated weight is 23 GiB
on a 1.7B model and the failure is an OOM twenty minutes in, after the model has
loaded. And it stamps the layer band into the artifact, because `density` is a
fraction of what was gated: 1% of seven layers and 1% of twenty-eight are
different claims and the number cannot tell them apart.

The result this is pointed at is not established. 5cc8dc3 reported a sheaf and
e4c92bb withdrew it -- the gates were sampled at evaluation, so the accuracy was
a random subnetwork's -- and after that fix the sweep is incoherent rather than
merely poor. The standing suspect is training scale, which is why `steps` and
`size` are the two flags with the loudest defaults here: the reference's unit is
an epoch and the run that produced the retracted number was 500 batches of 8.

This takes argparse where its neighbours take `sys.argv[1]`, because a script
with six tunables addressed by position is a script that gets run wrong. The
convention it does keep is the one that matters: the config is the first
argument, so moving from the 1.7B to the 8B is that argument and nothing else.

A common pipe could be: config | task | prune | budget | sheaf

Run: uv run python -m scripts.sheaf_prune qwen3-1.7b
     uv run python -m scripts.sheaf_prune qwen3-1.7b --layers 21-27 --steps 2000
     uv run python -m scripts.sheaf_prune gpt2-small --task ioi --steps 50
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.observe import banner, log, set_log_file, step
from scripts.paths import guard, result
from src.data.tasks import build_task, task_names
from src.methods.sheaves import gateable, load_bearing, prune, span
from src.model.adapter import load_adapter
from src.telemetry.journal import Journal, env_root, run_id
from src.telemetry.tracking import Tracker, load_tracking

# gate logit, gradient, and AdamW's two moments, all float32 and all one per
# gated weight. The originals are kept too, at the model's own dtype, which is
# the fifth term and the only one that is not four bytes.
STATE_BYTES_PER_GATE = 4 * 4

# What the autograd graph costs on top of the state, as a fixed part plus a
# per-gate part. It is not proportional to the gate count, which is what the
# first two versions of this assumed and why both were wrong: a multiplier
# fitted on a 151M-gate band said 39.6 B/gate and projected 79 GiB for the
# whole model, which then ran at 37.1. The caching allocator reuses blocks
# across parameter tensors, so most of the small-band peak is a constant --
# activations, the model's own forward buffers, allocator slack -- and only the
# tensors gumbel retains for the backward pass actually scale.
#
# Fitted on the two runs that have been measured here, and it reproduces both
# exactly: 151M gates peaked at 11.3 GiB and 1409M at 37.1. Rounded up from
# 5.00 and 4.02 for margin, so the projection sits a little above each.
#
# Both numbers come from torch's peak counter, which misses the driver's own
# allocations -- half of why a run once cleared a check and took
# NV_ERR_NO_MEMORY anyway. The other half was MemAvailable, fixed separately.
GRAPH_BASE_GIB = 6.0
GRAPH_BYTES_PER_GATE = 5

def budget(n_gates: int, weight_bytes: int, model_bytes: int) -> dict:
    """What the run costs before it starts: the model, the state, one graph

    Reported rather than discovered. The run loads the model first, so a band
    that does not fit fails after the expensive part rather than before it --
    and on a unified-memory host it fails as a driver error with no traceback,
    which reads as a broken script rather than as a band that was too wide.
    """
    optimizer = n_gates * STATE_BYTES_PER_GATE
    originals = n_gates * weight_bytes
    state = (optimizer + originals) / 2 ** 30
    model = model_bytes / 2 ** 30
    graph = GRAPH_BASE_GIB + n_gates * GRAPH_BYTES_PER_GATE / 2 ** 30
    return {
        "gates": n_gates,
        "optimizer_gib": round(optimizer / 2 ** 30, 2),
        "originals_gib": round(originals / 2 ** 30, 2),
        "model_gib": round(model, 2),
        "graph_gib": round(graph, 2),
        "total_gib": round(state, 2),
        "peak_gib": round(model + state + graph, 2),
    }

def available_gib() -> float:
    """Host memory this run can actually have, or 0 where that cannot be read

    Unified memory on the GB10 means the accelerator and the host draw on one
    pool, so the host figure is the binding one and `torch.cuda.mem_get_info`
    would report the same pool twice.

    MemFree rather than MemAvailable, which is the more pessimistic of the two
    on purpose. MemAvailable adds the page cache the kernel would reclaim under
    pressure, and a CUDA allocation does not get to wait for that: a run once
    read 27 GiB available here and took NV_ERR_NO_MEMORY from the driver two
    seconds later. The number that matters is what is free right now.
    """
    try:
        with Path("/proc/meminfo").open() as handle:
            for line in handle:
                if line.startswith("MemFree:"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 0.0

def parse_layers(text: str) -> list:
    """`21-27`, `21,23,26` or `all` into layer indices"""
    if text in ("", "all"):
        return None
    chosen = set()
    for piece in text.split(","):
        if "-" in piece.strip("-"):
            low, high = piece.split("-", 1)
            chosen.update(range(int(low), int(high) + 1))
        else:
            chosen.add(int(piece))
    return sorted(chosen)

def run(args: argparse.Namespace) -> None:
    set_log_file(result(f"sheaf-{args.task}.log"))
    with step("load model"):
        adapter = load_adapter(args.config)
    layers = parse_layers(args.layers)
    targets = gateable(adapter, layers)
    n_gates = sum(parameter.numel() for parameter in targets.values())
    weight_bytes = next(iter(targets.values())).element_size()
    model_bytes = sum(parameter.numel() * parameter.element_size()
                      for parameter in adapter.model.parameters())
    cost = budget(n_gates, weight_bytes, model_bytes)
    free = available_gib()

    with step(f"build '{args.task}'"):
        task = build_task(args.task, adapter, size=args.size, seed=args.seed)

    banner("discogp weight pruning", {
        "config": args.config,
        # Distinct prompts, not rows. The split keeps rows sharing a prompt
        # together, so the row count is not what the holdout is taken out of --
        # this task repeats a small pool, and reporting 128 here read as 128
        # independent examples when there were 25.
        "task": f"{args.task}, {len(set(task.clean))} distinct of {args.size} rows, "
                f"{1 - args.holdout:.0%}/{args.holdout:.0%} split by prompt",
        "band": "all layers" if layers is None else f"layers {span(layers)} of {len(adapter.blocks)}",
        "gates": f"{n_gates / 1e6:.0f}M",
        "state": f"{cost['total_gib']} GiB state, ~{cost['peak_gib']} GiB peak, "
                 f"{free:.0f} GiB available",
        "schedule": f"{args.steps} steps of {args.batch}, price {args.sparsity} x{args.max_times}",
        "artifact": result(f"sheaf-{args.task}.json"),
    })

    # The failure this replaces is an OOM after the model has loaded and the
    # task has been built, which reads as a broken script rather than as a band
    # that was too wide. Naming the smaller band is the whole point of saying so.
    if free and cost["peak_gib"] > free - args.reserve and not args.force:
        raise SystemExit(
            f"gating {n_gates / 1e6:.0f}M weights costs {cost['total_gib']} GiB of state and peaks "
            f"near {cost['peak_gib']} GiB with the graph, against {free:.0f} GiB available "
            f"(holding {args.reserve} GiB back).\n\n"
            f"  --layers 21-27   gates a band instead of the model, and says so in the artifact\n"
            f"  --force          runs it anyway\n\n"
            f"Freeing the pool is the other way: a serving process on this host holds most of it."
        )

    # A band that can be deleted outright without moving the metric cannot host
    # a circuit for it, and pruning inside one costs the sparsity term nothing.
    # This is measured, not assumed: layers 21,23,24,26 scored 1.000 shut and
    # went on to report 0.0008% density at accuracy 1.000 after 24 minutes.
    split = max(1, int(args.size * (1.0 - args.holdout)))
    with step("band control") as facts:
        control = load_bearing(adapter, task, layers, rows=range(split, args.size))
        facts["open"] = f"{control['open']:.3f}"
        facts["shut"] = f"{control['shut']:.3f}"
    if control["open"] - control["shut"] < args.needs and not args.force:
        raise SystemExit(
            f"shutting every gate in {'the model' if layers is None else f'layers {span(layers)}'} "
            f"moves held-out accuracy from {control['open']:.3f} to {control['shut']:.3f}, a drop of "
            f"{control['open'] - control['shut']:.3f}. The task survives without those weights, so "
            f"sparsity can close all of them for free and the run will report a density near zero "
            f"at full accuracy -- which looks like a circuit and is the band being irrelevant.\n\n"
            f"  --layers all     gates the whole model, where closing it does cost something\n"
            f"  --task ioi       a task this model does not already saturate\n"
            f"  --needs 0        runs it anyway and records the control in the artifact\n"
        )

    # The journal is opened before the loop and closed in a `finally`, so a run
    # killed mid-flight leaves its curve and a status of `failed` rather than
    # nothing. Both of the ways this has died -- the driver at two seconds, a
    # stale split at ninety minutes -- produced exactly nothing to look at.
    directory = env_root() / run_id(f"{args.config}-{args.task}")
    journal = Journal(directory, name=f"{args.config}-{args.task}", params={
        "config": args.config, "task": args.task, "layers": layers,
        "n_layers": len(adapter.blocks), "gates": n_gates, "steps": args.steps,
        "size": args.size, "distinct_prompts": len(set(task.clean)), "seed": args.seed,
        "batch": args.batch, "rate": args.rate, "sparsity": args.sparsity,
        "completeness": args.completeness, "max_times": args.max_times,
        "holdout": args.holdout, "band_control": control, "budget": cost,
    })
    log(f"journal: {directory} (tail -f {journal.metrics_path})")

    # The journal is the record and the tracker is a mirror, in that order: the
    # tracker is constructed after the journal and can fail without taking
    # anything with it. `active` false here means tracking is off or the server
    # did not answer, and the run proceeds either way.
    tracking = load_tracking(args.tracking)
    tracker = Tracker(tracking, name=directory.name, params=journal.params)
    if tracker.active:
        log(f"mlflow: {tracking.uri} experiment '{tracking.experiment}' run {tracker.run_id}")
    elif tracking.enabled:
        log(f"mlflow: DISABLED for this run -- {tracker.failure}")
    journal.sink = tracker
    try:
        with step(f"prune {args.steps} steps") as facts:
            sheaf = prune(
                adapter, task, steps=args.steps, rate=args.rate, sparsity=args.sparsity,
                completeness=args.completeness, batch=args.batch, max_times=args.max_times,
                holdout=args.holdout, layers=layers, journal=journal,
                probe_every=args.probe_every, seed=args.seed,
            )
            facts["density"] = f"{sheaf.density:.4%}"
            facts["held-out"] = f"{sheaf.accuracy:.3f}"
    except BaseException as error:
        journal.finish("failed", error=f"{type(error).__name__}: {error}")
        tracker.finish("FAILED")
        raise
    summary = {
        "density": sheaf.density, "accuracy": sheaf.accuracy,
        "train_accuracy": sheaf.train_accuracy,
        "complement_accuracy": sheaf.complement_accuracy,
        "baseline_accuracy": sheaf.baseline_accuracy,
    }
    journal.finish("completed", **summary)
    tracker.finish("FINISHED", summary=summary)

    log(str(sheaf))
    artifact = result(f"sheaf-{args.task}.json")
    artifact.write_text(json.dumps({
        "protocol": "DiscoGP (2407.03779) joint weight pruning: a gate per weight, the weights "
                    "frozen, the gates trained on faith + sparsity + completeness. `density` is "
                    "the fraction of *gated* weights left open, so it is only comparable across "
                    "runs that gated the same band.",
        "config": args.config,
        "task": args.task,
        "layers": sheaf.layers,
        "n_layers": len(adapter.blocks),
        "gates": sheaf.n_parameters,
        "open": sheaf.n_open,
        "density": round(sheaf.density, 6),
        "accuracy": round(sheaf.accuracy, 4),
        "train_accuracy": round(sheaf.train_accuracy, 4),
        "complement_accuracy": round(sheaf.complement_accuracy, 4),
        "baseline_accuracy": round(sheaf.baseline_accuracy, 4),
        "band_control": {key: round(value, 4) for key, value in control.items()},
        "settings": {
            "size": args.size, "seed": args.seed, "steps": args.steps, "batch": args.batch,
            "rate": args.rate, "sparsity": args.sparsity, "completeness": args.completeness,
            "max_times": args.max_times, "holdout": args.holdout,
        },
        "budget": cost,
        "history": sheaf.history,
        "standing": "the method is not reproduced here. 5cc8dc3's numbers were withdrawn by "
                    "e4c92bb (gates were sampled at evaluation, so accuracy never depended on "
                    "them); with that fixed the density/accuracy relation is incoherent rather "
                    "than poor. Read this artifact as a run of the method, not as a result from "
                    "it, until a sweep is monotone in density.",
        "journal": str(directory),
        "command": f"uv run python -m scripts.sheaf_prune {args.config} --task {args.task}"
                   f" --layers {args.layers} --steps {args.steps} --size {args.size}",
    }, indent=2) + "\n")
    if args.save_gates:
        torch.save({name: logits.cpu() for name, logits in sheaf.gates.items()},
                   result(f"sheaf-{args.task}-gates.pt"))
        log(f"-> {result(f'sheaf-{args.task}-gates.pt')}")
    log(f"-> {artifact}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default="qwen3-1.7b",
                        help="model config; the 1.7B, because the 8B's gates do not fit")
    parser.add_argument("--task", default="translation", choices=task_names())
    parser.add_argument("--layers", default="all", help="'all', '21-27' or '21,23,26'")
    parser.add_argument("--size", type=int, default=128,
                        help="prompts; the retracted run had 32 and data scale is the standing suspect")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000,
                        help="the reference's unit is an epoch; the retracted run was 500 batches")
    # 64, not 8. Measured on gpt2-small: batch 8 runs 0.428 s/step and 18.7
    # prompts/s; batch 128 runs 0.746 s/step and 171.5 -- 9.2x the throughput
    # for 1.74x the step. Batch 16 is free outright. The gate machinery alone
    # costs 0.087 s/step, so the rest is two masked forward passes that at
    # batch 8 are launch-latency bound rather than compute bound. The graph is
    # per-weight, which is why shrinking the batch never bought memory and
    # growing it barely costs time -- the same fact read in both directions.
    # Every result before 2026-09-02 was taken at 8 and is not comparable
    # step-for-step with one taken here.
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--sparsity", type=float, default=1.0, help="the starting price")
    parser.add_argument("--max-times", type=float, default=1000.0, dest="max_times",
                        help="the factor the price ramps to; the reference's default")
    parser.add_argument("--completeness", type=float, default=0.3)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--reserve", type=float, default=4.0,
                        help="GiB of headroom held back on top of the projected peak")
    parser.add_argument("--needs", type=float, default=0.05,
                        help="accuracy the band must cost when shut, or the run is refused")
    parser.add_argument("--tracking", default="none",
                        help="tracking config in configs/tracking/ ('mlflow'), or 'none'")
    parser.add_argument("--probe-every", type=int, default=10, dest="probe_every",
                        help="steps between hard-density probes; 0 logs the loss terms only")
    parser.add_argument("--force", action="store_true",
                        help="run past the memory and band-control checks")
    # On by default, because the mask IS the result. It was opt-in once, and a
    # five-point sweep ran without it: seventeen minutes a point, five masks
    # discarded, and the summary JSON kept. The best of them scored 0.961 held
    # out at 0.33% density and cannot be recovered -- torch's RNG was unseeded
    # then too, so rerunning produces a different mask rather than that one.
    parser.add_argument("--no-save-gates", action="store_false", dest="save_gates",
                        help="skip writing the learned mask; it is the run's actual product")
    parser.set_defaults(save_gates=True)
    args = parser.parse_args()
    guard(args.config)
    run(args)

if __name__ == "__main__":
    main()
