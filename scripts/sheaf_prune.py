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

A common pipe could be: parse_layers | run_budget | build_task | load_bearing | prune | sheaf

Run: uv run python -m scripts.sheaf_prune qwen3-1.7b
     uv run python -m scripts.sheaf_prune qwen3-1.7b --layers 21-27 --steps 2000
     uv run python -m scripts.sheaf_prune gpt2-small --task ioi --steps 50
"""

import argparse
import json
import sys

import torch

from src.data.tasks import build_task, task_names
from src.methods.gates import GATES_FILE, MASK_FILE, pack, parse_layers, run_budget
from src.methods.sheaves import load_bearing, prune, span
from src.model.adapter import load_adapter
from src.telemetry.journal import Journal, env_root, run_id
from src.telemetry.observe import banner, host_memory_gib, log, set_log_file, step
from src.telemetry.results import guard, result
from src.telemetry.tracking import Tracker, load_tracking


def run(args: argparse.Namespace) -> None:
    set_log_file(result(f"sheaf-{args.task}.log"))
    with step("load model"):
        adapter = load_adapter(args.config)
    layers = parse_layers(args.layers)
    # the budget is reported before anything else is allocated: the failure it
    # replaces is an OOM after the model has loaded, which on a unified-memory
    # host is a driver error with no traceback
    cost = run_budget(adapter, layers)
    n_gates = cost["gates"]
    # MemFree, not MemAvailable: a CUDA allocation cannot wait for the page cache
    free = host_memory_gib("MemFree")

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
        "edge_sparsity": args.edge_sparsity, "faith": args.faith_kind,
        "init": args.init, "temperature": args.temperature, "anneal": args.anneal,
        "target": args.target, "protect": args.protect, "warmup": args.warmup,
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
                edge_sparsity=args.edge_sparsity, faith_kind=args.faith_kind,
                init=args.init, temperature=args.temperature, anneal=args.anneal,
                target=args.target, protect=args.protect, warmup=args.warmup,
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
        "n_edges": sheaf.n_edges,
        "n_edges_open": sheaf.n_edges_open,
        "n_pinned": sheaf.n_pinned,
        "edge_density": (None if sheaf.edge_density is None else round(sheaf.edge_density, 6)),
        "accuracy": round(sheaf.accuracy, 4),
        "train_accuracy": round(sheaf.train_accuracy, 4),
        "complement_accuracy": round(sheaf.complement_accuracy, 4),
        "baseline_accuracy": round(sheaf.baseline_accuracy, 4),
        "band_control": {key: round(value, 4) for key, value in control.items()},
        "settings": {
            "size": args.size, "seed": args.seed, "steps": args.steps, "batch": args.batch,
            "rate": args.rate, "sparsity": args.sparsity, "completeness": args.completeness,
            "max_times": args.max_times, "holdout": args.holdout,
            "edge_sparsity": args.edge_sparsity, "faith": args.faith_kind,
            "init": args.init, "temperature": args.temperature, "anneal": args.anneal,
            "target": args.target, "protect": args.protect, "warmup": args.warmup,
        },
        "budget": cost,
        "history": sheaf.history,
        "standing": "the method is not reproduced here. 5cc8dc3's numbers were withdrawn by "
                    "e4c92bb (gates were sampled at evaluation, so accuracy never depended on "
                    "them); with that fixed the density/accuracy relation is incoherent rather "
                    "than poor. Read this artifact as a run of the method, not as a result from "
                    "it, until a sweep is monotone in density.",
        "journal": str(directory),
        "command": " ".join(["uv run python -m scripts.sheaf_prune", *sys.argv[1:]]),
    }, indent=2) + "\n")
    # The mask is the run's product and is always written: one bit per gate,
    # small enough to copy off the box. The logits are 32x that and are the
    # optimizer's state, not the circuit; kept only when asked for.
    torch.save(pack(sheaf.gates), result(MASK_FILE.format(task=args.task)))
    log(f"-> {result(MASK_FILE.format(task=args.task))}")
    if args.save_gates:
        torch.save({name: logits.cpu() for name, logits in sheaf.gates.items()},
                   result(GATES_FILE.format(task=args.task)))
        log(f"-> {result(GATES_FILE.format(task=args.task))}")
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
    parser.add_argument("--init", type=float, default=1.0,
                        help="the starting gate logit; 1.0 is the reference's, and samples 27%% of the "
                             "weights shut at step 0, which a 1.7B does not survive. 5.0 is 0.7%%.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="backward sharpness of the gate, never which gates open; the reference "
                             "trains weight masks at 0.01")
    parser.add_argument("--target", type=float, default=None,
                        help="a density to hold instead of a price to guess: the price is learned "
                             "(Wang et al. 2020 / CoFi Lagrangian); --sparsity and --max-times are "
                             "ignored, and --warmup defaults to half the run")
    parser.add_argument("--warmup", type=int, default=None,
                        help="steps the price ramps over (default: the whole run) or, with "
                             "--target, the density ramps over (default: half of it). Longer "
                             "than the run is allowed: a 500-step run with --warmup 1000 walks "
                             "the first half of a 2000-step run's target schedule")
    parser.add_argument("--protect", type=float, default=0.0,
                        help="pin the top fraction of weights by |w| open, outside the search: "
                             "the first-order faith gradient underprices closing them 25x on "
                             "Qwen3-1.7B, and every run at any price went to chance where they shut")
    parser.add_argument("--anneal", action="store_true",
                        help="shrink the gate noise to zero across the run, so the mask trained "
                             "last is the thresholded one that is saved")
    parser.add_argument("--sparsity", type=float, default=1.0, help="the starting price")
    parser.add_argument("--max-times", type=float, default=1000.0, dest="max_times",
                        help="the factor the price ramps to; the reference's default")
    # The other half of DiscoGP: it prunes edges and weights jointly, and this
    # was weights-only until the weights-only mask was shown to rank at 0.938
    # while generating ' Mary Emma Rose the Rose'. Priced separately because
    # ~2k edges against 85M weights would otherwise be numerically invisible.
    # 0 disables it and the run is exactly the weights-only one.
    parser.add_argument("--edge-sparsity", type=float, default=0.0, dest="edge_sparsity",
                        help="price on open edges; 0 prunes weights only, as before")
    # "pair" is the reference's term and the one measured to be too weak: it
    # certified a circuit that ranks at 0.938 on unseen prompts and generates
    # ' Mary Emma Rose the Rose'. "kl" scores the whole last-token distribution
    # against the unmasked model's, which a ranking shortcut cannot satisfy.
    parser.add_argument("--faith", default="pair", choices=("pair", "kl", "nll"), dest="faith_kind",
                        help="faithfulness on the good/bad pair, or KL over the distribution")
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
    # The mask IS the result and is always written (see the end of `run`). It
    # was opt-in once, and a five-point sweep ran without it: seventeen
    # minutes a point, five masks discarded, and the summary JSON kept. The
    # best of them scored 0.961 held out at 0.33% density and cannot be
    # recovered -- torch's RNG was unseeded then too, so rerunning produces a
    # different mask rather than that one. The float logits are the extra:
    # 5.6 GB on the whole 1.7B, and nothing downstream reads past their sign.
    parser.add_argument("--save-gates", action="store_true", dest="save_gates",
                        help="also write the float gate logits beside the packed mask")
    args = parser.parse_args()
    guard(args.config)
    run(args)

if __name__ == "__main__":
    main()
