"""The logit-lens baseline on WorkspaceBench: the floor every other readout beats.

WorkspaceBench asks whether an activation-reading lens surfaces what the model
computed and never wrote. Before any trained lens is worth fitting, the number
to have is what the untrained one gets, because a logit lens costs one forward
pass: if an NLA does not beat this, the training bought nothing.

Only the banks a one-token readout can be asked are run -- `readable()` keeps
the items naming a prompt, intermediates and a single-token read position, and
drops the sixteen banks that want prose back. Each item is read at every layer
and the model is also asked to answer it, because the benchmark's premise is
that the intermediate is computed and not written: where the answer contains
it anyway, a hit cannot be told from the model having said the word, and those
items are excluded from `clean_hit_rate`.

Resumable per bank, since the whole set on a 27B is hours: a bank already in
the results file is skipped.

Tracked in MLflow under `mi-lab-wsbench` like every run here: one row of
metrics per bank as it lands (step = the bank's position in the run), the
totals at the end, and the results file and log uploaded. `--backfill`
sends a results file that was written before tracking existed, loading no
model.

Run: uv run python -m scripts.wsbench_baseline <config> [bank ...]
     uv run python -m scripts.wsbench_baseline qwen3.6-27b association multihop
     uv run python -m scripts.wsbench_baseline qwen3-1.7b --backfill
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from src.data.workspacebench import SINGLE_TOKEN_BANKS, WorkspaceBench, readable
from src.methods.readout import report, score_bank
from src.model.adapter import load_adapter
from src.telemetry.observe import banner, duration, gpu, log
from src.telemetry.tracking import Tracker, track

RESULTS = Path("results/wsbench")
EXPERIMENT = "mi-lab-wsbench"

# The per-bank numbers MLflow charts; `hit_rate_by_layer` stays in the file
BANK_METRICS = ("items", "hit_rate", "clean_hit_rate", "leak_rate", "degeneracy", "median_first_hit_layer", "seconds")

def results_path(config: str) -> Path:
    return RESULTS / f"logit-lens-{config}.json"

def load(config: str) -> Dict:
    path = results_path(config)
    return json.loads(path.read_text()) if path.exists() else {"config": config, "banks": {}}

def save(config: str, state: Dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    results_path(config).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

def send_bank(tracker: Tracker, step: int, family: str, summary: Dict) -> None:
    """One bank's numbers as one step, each metric prefixed with the bank so the charts separate them"""
    tracker.log(step, {f"{family}.{key}": summary.get(key) for key in BANK_METRICS})
    tracker.flush()

def totals(state: Dict) -> Dict:
    """Items and hit rate over every bank, weighted by bank size"""
    banks = state["banks"]
    count = sum(held["summary"]["items"] for held in banks.values())
    hits = sum(held["summary"]["hit_rate"] * held["summary"]["items"] for held in banks.values())
    return {"items": count, "hit_rate": hits / count if count else 0.0, "banks": len(banks)}

def run(config: str, families: List[str], limit: int = 0) -> None:
    with track(f"logit-lens-{config}", EXPERIMENT, outputs=[results_path(config), log_path(config)],
               params={"config": config, "banks": ",".join(families), "limit": limit, "top_k": 10},
               tags={"script": "wsbench_baseline", "method": "logit-lens"}) as tracker:
        scored(config, families, limit, tracker)

def scored(config: str, families: List[str], limit: int, tracker: Tracker) -> None:
    state = load(config)
    adapter = load_adapter(config)
    tracker.log_params({"model": adapter.cfg.hf_name, "n_layers": adapter.cfg.n_layers,
                        "d_model": adapter.cfg.d_model, "batch_size": adapter.cfg.batch_size,
                        "dtype": adapter.cfg.dtype, "max_new_tokens": adapter.cfg.max_new_tokens})
    banner("logit lens on WorkspaceBench", {
        "model": f"{adapter.cfg.hf_name} ({adapter.cfg.n_layers} layers, d_model {adapter.cfg.d_model})",
        "banks": len(families),
        "batch": adapter.cfg.batch_size,
        "writing": results_path(config),
    })

    for family in families:
        if family in state["banks"]:
            log(f"{family}: already done, skipping", indent=1)
            continue
        bank = WorkspaceBench(family)
        items = readable(bank.items)[: limit or None]
        if not items:
            log(f"{family}: no items a one-token readout can be asked, skipping", indent=1)
            continue

        started = time.perf_counter()
        scores = score_bank(adapter, items, family=family, top_k=10)
        summary = report(scores)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        state["banks"][family] = {"summary": summary, "gate": bank.gate,
                                  "items": [asdict(score) for score in scores]}
        save(config, state)
        send_bank(tracker, families.index(family), family, summary)

        log(f"{family}: {summary['items']} items in {duration(summary['seconds'])} · {gpu()}", indent=1)
        log(f"hit {summary['hit_rate']:.1%} · clean {summary['clean_hit_rate']:.1%} · "
            f"leaked {summary['leak_rate']:.1%} · first hit at layer {summary['median_first_hit_layer']} · "
            f"degeneracy {summary['degeneracy']:.1%}", indent=2)

    overall(state)
    if state["banks"]:
        tracker.log(len(families), {f"all.{key}": value for key, value in totals(state).items()})
        tracker.flush()
    log(f"-> {results_path(config)}")

def log_path(config: str) -> Path:
    """Where a backgrounded run's stdout is sent by convention, uploaded if it exists"""
    return RESULTS / f"logit-lens-{config}.log"

def backfill(config: str) -> None:
    """Send a results file written before this script tracked, without loading the model

    The run is tagged `backfill` so it is not mistaken for a fresh
    measurement: its provenance is the commit that *sent* it, not the one
    that computed it, and its timestamps are today's.
    """
    state = load(config)
    if not state["banks"]:
        raise SystemExit(f"no results at {results_path(config)} to backfill")
    families = sorted(state["banks"])
    # no `outputs`: those upload what changed during the block, and a backfilled file changed long before it
    with track(f"logit-lens-{config}", EXPERIMENT,
               params={"config": config, "banks": ",".join(families), "top_k": 10},
               tags={"script": "wsbench_baseline", "method": "logit-lens", "backfill": "true"}) as tracker:
        for step, family in enumerate(families):
            send_bank(tracker, step, family, state["banks"][family]["summary"])
        tracker.log(len(families), {f"all.{key}": value for key, value in totals(state).items()})
        tracker.flush()
        for path in (results_path(config), log_path(config)):
            tracker.log_artifact(path, name=str(path))
        log(f"backfilled {len(families)} banks of {config}: {tracker.url or tracker.failure or 'tracking off'}")

def overall(state: Dict) -> None:
    """Every bank on one line, and the total underneath"""
    banks = state["banks"]
    if not banks:
        return
    log("=" * 78)
    log(f"{'bank':<24} {'n':>5} {'hit':>7} {'clean':>7} {'leaked':>7} {'layer':>6}")
    for family, held in sorted(banks.items()):
        summary = held["summary"]
        clean = summary["clean_hit_rate"]
        log(f"{family:<24} {summary['items']:>5} {summary['hit_rate']:>6.1%} "
            f"{clean if clean is None else format(clean, '>6.1%')} {summary['leak_rate']:>6.1%} "
            f"{summary['median_first_hit_layer']!s:>6}")
    total = sum(held["summary"]["items"] for held in banks.values())
    hits = sum(held["summary"]["hit_rate"] * held["summary"]["items"] for held in banks.values())
    log(f"{'all':<24} {total:>5} {hits / total:>6.1%}")

def main() -> None:
    """<config> [limit] [bank ...] -- a bare number after the config caps items per bank; --backfill sends a file"""
    args = sys.argv[1:]
    config = args.pop(0) if args else "qwen3.6-27b"
    if "--backfill" in args:
        backfill(config)
        return
    limit = int(args.pop(0)) if args and args[0].isdigit() else 0
    run(config, args or list(SINGLE_TOKEN_BANKS), limit=limit)

if __name__ == "__main__":
    main()
