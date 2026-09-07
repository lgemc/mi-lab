"""Run DiscoGP weight pruning on a task, which until now had no way to be run.

`methods/sheaves/` has been exercised only from the test suite and a REPL, on
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
import os
import sys
from pathlib import Path

import torch

from src.data.tasks import build_task, task_names
from src.experiment.sheaf import SheafError, SheafSpec
from src.methods.sheaves.gateable import span
from src.methods.sheaves.mask import BEST_MASK_FILE, GATES_FILE, MASK_FILE, UNITS_FILE, pack, parse_layers, run_budget
from src.methods.sheaves.training import load_bearing, prune
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
        "gates": (f"{len(adapter.edges())} edges, weights ungated" if args.edges_only
                  else f"{n_gates / 1e6:.0f}M"),
        "state": f"{cost['total_gib']} GiB state, ~{cost['peak_gib']} GiB peak, "
                 f"{free:.0f} GiB available",
        "schedule": f"{args.steps} steps of {args.batch}, price {args.sparsity} x{args.max_times}",
        "artifact": result(f"sheaf-{args.task}.json"),
    })

    # The failure this replaces is an OOM after the model has loaded and the
    # task has been built, which reads as a broken script rather than as a band
    # that was too wide. Naming the smaller band is the whole point of saying so.
    # An edges-only run has no weight gates, so none of this state exists: the
    # 23 GiB is four float32 tensors per *gated weight* and there are none.
    if args.edges_only:
        pass
    elif free and cost["peak_gib"] > free - args.reserve and not args.force:
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
        "dual_rate": args.dual_rate, "dual_restart": args.dual_restart,
        "granular": args.granular, "attribute": args.attribute, "init_low": args.init_low,
        # How the accuracy curve was measured is part of what the curve means:
        # the same run probed on 8 rows and on 128 produces two different
        # pictures of itself, and only one of them is readable. These were
        # missing, so MLflow could show the curve and not its denominator.
        "probe_every": args.probe_every, "probe_size": args.probe_size,
        "edges_only": args.edges_only, "results": args.results,
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
                probe_every=args.probe_every, probe_size=args.probe_size, seed=args.seed,
                edge_sparsity=args.edge_sparsity, faith_kind=args.faith_kind,
                init=args.init, temperature=args.temperature, anneal=args.anneal,
                target=args.target, protect=args.protect, warmup=args.warmup,
                dual_rate=args.dual_rate, dual_restart=args.dual_restart,
                granular=args.granular.split(",") if args.granular else None,
                attribute=args.attribute, init_low=args.init_low,
                gate_weights=not args.edges_only,
            )
            # The metric's floor, measured above and not assumed: `accuracy` is
            # a two-way comparison and `recovered` is the only reading of it
            # that survives a floor at 0.477.
            sheaf.chance = control["shut"]
            facts["density"] = (f"{sheaf.edge_density:.4%} of edges" if args.edges_only
                                else f"{sheaf.density:.4%}")
            facts["held-out"] = f"{sheaf.accuracy:.3f}"
            if sheaf.recovered is not None:
                facts["recovered"] = f"{sheaf.recovered:.3f}"
            facts["first token"] = f"{sheaf.first_token:.3f}"
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
        "protocol": ("edge pruning after DiscoGP (2407.03779) with the weights left whole: a "
                     "gate per (source, destination) edge of the residual stream, trained on "
                     "faith + sparsity"
                     + (" + completeness" if args.completeness else "")
                     + " (see `settings`). `density` is 100% by construction and "
                     "`edge_density` is the circuit; `recovered` rescales `accuracy` onto the "
                     "range between the shut band and the full model."
                     if args.edges_only else
                     "DiscoGP (2407.03779) joint weight pruning: a gate per weight, the weights "
                     "frozen, the gates trained on faith + sparsity + completeness. `density` is "
                     "the fraction of *gated* weights left open, so it is only comparable across "
                     "runs that gated the same band."),
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
        "n_restarts": sheaf.n_restarts,
        "edge_density": (None if sheaf.edge_density is None else round(sheaf.edge_density, 6)),
        "accuracy": round(sheaf.accuracy, 4),
        # The last mask is `accuracy`; the best one the run ever held is this,
        # rescored on the same held-out rows. They come apart when the step
        # budget runs out somewhere other than the bottom of the curve.
        "best_accuracy": (None if sheaf.best_accuracy is None else round(sheaf.best_accuracy, 4)),
        "best_density": (None if sheaf.best_density is None else round(sheaf.best_density, 6)),
        "best_step": sheaf.best_step,
        "best_edges_open": (None if sheaf.best_edges is None
                            else [list(edge) for edge in sheaf.best_edges]),
        # The ranking rescaled onto the range it can actually move over. The
        # sweep's raw numbers (0.61-0.72) sit above a floor of 0.477 and were
        # read as if the floor were zero, which doubled every result in the
        # table. This is the number to compare runs on.
        "recovered": (None if sheaf.recovered is None else round(sheaf.recovered, 4)),
        "first_token": round(sheaf.first_token, 4),
        "baseline_first_token": round(sheaf.baseline_first_token, 4),
        "train_accuracy": round(sheaf.train_accuracy, 4),
        "complement_accuracy": round(sheaf.complement_accuracy, 4),
        "baseline_accuracy": round(sheaf.baseline_accuracy, 4),
        "band_control": {key: round(value, 4) for key, value in control.items()},
        "settings": {
            "size": args.size, "seed": args.seed, "probe_size": args.probe_size,
            "steps": args.steps, "batch": args.batch,
            "rate": args.rate, "sparsity": args.sparsity, "completeness": args.completeness,
            "max_times": args.max_times, "holdout": args.holdout,
            "edge_sparsity": args.edge_sparsity, "faith": args.faith_kind,
            "init": args.init, "temperature": args.temperature, "anneal": args.anneal,
            "target": args.target, "protect": args.protect, "warmup": args.warmup,
            "dual_rate": args.dual_rate, "dual_restart": args.dual_restart,
            "granular": args.granular, "attribute": args.attribute, "init_low": args.init_low,
        },
        "units": sheaf.units,
        # The circuit itself when the run pruned edges: `n_edges_open` is a
        # count and the (source, destination) list is the result.
        "edges_open": (None if sheaf.edges is None
                       else [list(edge) for edge in sheaf.edges]),
        "budget": cost,
        "history": sheaf.history,
        "standing": "the method is not reproduced here. 5cc8dc3's numbers were withdrawn by "
                    "e4c92bb (gates were sampled at evaluation, so accuracy never depended on "
                    "them); with that fixed the density/accuracy relation is incoherent rather "
                    "than poor. Read this artifact as a run of the method, not as a result from "
                    "it, until a sweep is monotone in density.",
        "journal": str(directory),
        "command": " ".join([f"uv run python -m scripts.{Path(sys.argv[0]).stem}",
                             *sys.argv[1:]]),
    }, indent=2) + "\n")
    # The mask is the run's product and is always written: one bit per gate,
    # small enough to copy off the box. The logits are 32x that and are the
    # optimizer's state, not the circuit; kept only when asked for.
    # An edges-only run has no weight mask, and a file of 1.4e9 ones is not a
    # circuit; its product is `edges_open` in the artifact above.
    if sheaf.gates:
        torch.save(pack(sheaf.gates), result(MASK_FILE.format(task=args.task)))
        log(f"-> {result(MASK_FILE.format(task=args.task))}")
    # Written beside the last mask, never instead of it: which one to serve is a
    # judgement (the best is sparser and scored better, the last is what the
    # schedule actually converged to) and deleting either would make it for you.
    if sheaf.best_gates and sheaf.best_step is not None:
        best_path = result(BEST_MASK_FILE.format(task=args.task))
        torch.save(pack(sheaf.best_gates), best_path)
        log(f"-> {best_path}")
    if args.save_gates:
        torch.save({name: logits.cpu() for name, logits in sheaf.gates.items()},
                   result(GATES_FILE.format(task=args.task)))
        log(f"-> {result(GATES_FILE.format(task=args.task))}")
    if sheaf.unit_logits is not None:
        # Small -- one float per head, neuron and block -- and the only record
        # of which units closed, since the mask folds them into the weights.
        torch.save(sheaf.unit_logits, result(UNITS_FILE.format(task=args.task)))
        log(f"-> {result(UNITS_FILE.format(task=args.task))}")
    log(f"-> {artifact}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", nargs="?", default="qwen3-1.7b",
                        help="model config; the 1.7B, because the 8B's gates do not fit")
    parser.add_argument("--task", default="translation", choices=task_names())
    parser.add_argument("--results", default=os.environ.get("MI_LAB_RESULTS"),
                        help="where the run writes; a field of the spec rather than only an "
                             "environment variable, because where a run writes is part of "
                             "what the run is")
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
    parser.add_argument("--dual-rate", type=float, default=None, dest="dual_rate",
                        help="with --target: ascend the price by plain gradient ascent at this rate, "
                             "proportional to the density gap, instead of through the gates' AdamW "
                             "(which moves it ~lr per step whatever the gap and overshot a 20%% "
                             "target to 12.6%% on the 1.7B)")
    parser.add_argument("--dual-restart", action="store_true", dest="dual_restart",
                        help="with --target: reset the price to zero on any step the mask is at or "
                             "under the target (Gallego-Posada et al. 2022), so a reached target "
                             "trains on faith alone")
    parser.add_argument("--warmup", type=int, default=None,
                        help="steps the price ramps over (default: the whole run) or, with "
                             "--target, the density ramps over (default: half of it). Longer "
                             "than the run is allowed: a 500-step run with --warmup 1000 walks "
                             "the first half of a 2000-step run's target schedule")
    parser.add_argument("--protect", type=float, default=0.0,
                        help="pin the top fraction of weights by |w| open, outside the search: "
                             "the first-order faith gradient underprices closing them 25x on "
                             "Qwen3-1.7B, and every run at any price went to chance where they shut")
    parser.add_argument("--granular", nargs="?", const="head,kv,neuron", default=None,
                        metavar="FAMILIES",
                        help="a gate per unit over the gate per weight, multiplied in (Haider et "
                             "al. COLM 2026); a unit the task can spare closes by one parameter. "
                             "Comma-separated families from head, kv, neuron, block; bare "
                             "--granular is head,kv,neuron (blocks are too coarse for the price "
                             "to track)")
    parser.add_argument("--attribute", type=int, default=0, metavar="BATCHES",
                        help="warm-start every gate from |w * dL/dw| on this many batches of the "
                             "full model: logits are the attribution rank mapped onto "
                             "[--init-low, --init]")
    parser.add_argument("--init-low", type=float, default=2.0,
                        help="the logit the least implicated weight of each tensor starts at "
                             "under --attribute")
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
    # The other direction, and the one the sweep argues for. A gate per weight
    # is 1.4e9 free bits against ~350 distinct prompts, and every run of the
    # sweep memorized (train 0.93-0.99, held out 0.61-0.72) at every density
    # and every faith kind -- which is what the strong lottery-ticket results
    # predict a mask with that much freedom will do. Edges are ~14k gates on
    # this model, five orders of magnitude smaller, and they are the unit the
    # word "circuit" refers to.
    parser.add_argument("--edges-only", action="store_true", dest="edges_only",
                        help="leave every weight open and prune edges alone; needs "
                             "--edge-sparsity or --target, and refuses --protect, "
                             "--granular and --attribute, which gate weights")
    # "pair" is the reference's term and the one measured to be too weak: it
    # certified a circuit that ranks at 0.938 on unseen prompts and generates
    # ' Mary Emma Rose the Rose'. "kl" scores the whole last-token distribution
    # against the unmasked model's, which a ranking shortcut cannot satisfy.
    parser.add_argument("--faith", default="pair", choices=("pair", "kl", "nll", "gold"),
                        dest="faith_kind",
                        help="faithfulness on the good/bad pair, KL over the distribution, nll on "
                             "the full model's argmax (the paper's), or nll on the task's answer")
    parser.add_argument("--completeness", type=float, default=0.3)
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--reserve", type=float, default=4.0,
                        help="GiB of headroom held back on top of the projected peak")
    parser.add_argument("--needs", type=float, default=0.05,
                        help="accuracy the band must cost when shut, or the run is refused")
    # On by default, and the journal is untouched by it. Every run in
    # results/qwen3-1.7b-sweep/ was launched on the old `none` default, so a
    # working MLflow server sat empty through the whole sweep and nobody could
    # see it was empty *because* nothing had been sent. The tracker is a mirror
    # (telemetry/tracking.py): the row is on disk before it is posted, and a
    # network failure disables the sink and says so once rather than raising
    # into a two-hour training loop -- so defaulting it on cannot cost a run.
    parser.add_argument("--tracking", default=os.environ.get("MI_LAB_TRACKING", "mlflow"),
                        help="tracking config in configs/tracking/ ('mlflow', the default), or "
                             "'none' to mirror nowhere; MI_LAB_TRACKING overrides. The journal "
                             "and the artifact are written either way")
    parser.add_argument("--probe-size", type=int, default=128, dest="probe_size",
                        help="held-out rows the in-run accuracy probe scores; 0 uses --batch, "
                             "which is what every run before this did and is +-0.18 at batch 8")
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
    # Through the same schema the YAML door uses, so a flag and a key cannot
    # validate differently. `sheaves/run/*.yaml` is the door to prefer -- a run
    # in a file can be read and diffed before it costs two hours -- and this one
    # stays because a one-off does not deserve a commit.
    try:
        spec = SheafSpec.from_mapping({
            "config": args.config, "task": args.task, "layers": args.layers,
            "results": args.results, "size": args.size, "seed": args.seed,
            "holdout": args.holdout, "steps": args.steps, "batch": args.batch,
            "rate": args.rate, "probe_every": args.probe_every, "probe_size": args.probe_size,
            "faith": args.faith_kind, "completeness": args.completeness,
            "init": args.init, "init_low": args.init_low,
            "temperature": args.temperature, "anneal": args.anneal,
            "protect": args.protect, "granular": args.granular,
            "attribute": args.attribute, "target": args.target,
            "warmup": args.warmup, "dual_rate": args.dual_rate,
            "dual_restart": args.dual_restart, "sparsity": args.sparsity,
            "max_times": args.max_times, "edges_only": args.edges_only,
            "edge_sparsity": args.edge_sparsity, "reserve": args.reserve,
            "needs": args.needs, "force": args.force,
            "save_gates": args.save_gates, "tracking": args.tracking,
        })
    except SheafError as error:
        raise SystemExit(str(error)) from None
    if spec.results:
        os.environ["MI_LAB_RESULTS"] = str(spec.results)
    guard(spec.config)
    run(spec)

if __name__ == "__main__":
    main()
