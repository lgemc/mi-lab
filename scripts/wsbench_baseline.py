"""Token readouts on WorkspaceBench, one method at a time, into one leaderboard.

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

Methods are the ones that emit a token and differ only in how a layer's
residual reaches the unembedding: `logit-lens` (not at all) and `jlens` (the
published Jacobian lens for the config, `LENSES`, or `--lens <file>`). Every
method is scored by the same `score_bank` on the same items, and each writes
`results/wsbench/<method>-<config>.json`; `wsbench_leaderboard` reads them all.
The model's own answers do not depend on the readout, so a second method on a
model reuses the first one's rather than generating them again.

Tracked in MLflow under `mi-lab-wsbench` like every run here: one row of
metrics per bank as it lands (step = the bank's position in the run), the
totals at the end, and the results file and log uploaded. `--backfill`
sends a results file that was written before tracking existed, loading no
model.

Run: uv run python -m scripts.wsbench_baseline <config> [--method M[,M...]] [--lens FILE] [limit] [bank ...]
     uv run python -m scripts.wsbench_baseline qwen3.6-27b association multihop
     uv run python -m scripts.wsbench_baseline qwen3.6-27b --method jlens-paired,rlens
     uv run python -m scripts.wsbench_baseline qwen3-1.7b --backfill
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.domains.lm.analysis.lens import JacobianLens, report, score_bank
from src.domains.lm.data.workspacebench import SINGLE_TOKEN_BANKS, WorkspaceBench, readable
from src.model.adapter import load_adapter
from src.telemetry.observe import banner, duration, gpu, log, set_log_file
from src.telemetry.tracking import Tracker, track

RESULTS = Path("results/wsbench")
EXPERIMENT = "mi-lab-wsbench"
# The published lens behind each method, per config: (repo, file) on the Hugging Face hub.
#
# `jlens` is the reference implementation's fit on wikitext (n=1000 for the 27B), the lens
# WorkspaceBench's own reference readouts come from. `jlens-paired` and `rlens` are a matched
# pair from the benchmark's authors -- same 25 pile-10k prompts, same target layer (n - 2), same
# forward pass -- differing only in the backward rules, so the two are the fair J-versus-R
# comparison and `jlens` against `jlens-paired` shows what the fitting recipe alone moves.
NEURONPEDIA = "neuronpedia/jacobian-lens"
WORKSPACE = "camilablank/workspace-lenses"
LENSES = {
    "jlens": {
        "qwen3.6-27b": (NEURONPEDIA, "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
        "qwen3-1.7b": (NEURONPEDIA, "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt"),
    },
    "jlens-paired": {"qwen3.6-27b": (WORKSPACE, "qwen3.6-27b/j-lens/lens.pt")},
    "rlens": {"qwen3.6-27b": (WORKSPACE, "qwen3.6-27b/r-lens/lens.pt")},
}
METHODS = ("logit-lens", *LENSES)

# The per-bank numbers MLflow charts; `hit_rate_by_layer` stays in the file
BANK_METRICS = ("items", "hit_rate", "clean_hit_rate", "leak_rate", "degeneracy", "median_first_hit_layer", "seconds")

def results_path(config: str, method: str = "logit-lens") -> Path:
    return RESULTS / f"{method}-{config}.json"

def log_path(config: str, method: str = "logit-lens") -> Path:
    """The method's own log, written by `run` and uploaded with its MLflow run"""
    return RESULTS / f"{method}-{config}.log"

def load(config: str, method: str = "logit-lens") -> Dict:
    path = results_path(config, method)
    state = json.loads(path.read_text()) if path.exists() else {"config": config, "banks": {}}
    state.setdefault("method", method)  # files written before there was more than one method
    return state

def save(state: Dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    results_path(state["config"], state["method"]).write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

def earlier_answers(config: str, method: str) -> Dict[str, Dict[str, str]]:
    """The model's answers, by bank and item, from any other method's run on this config"""
    found: Dict[str, Dict[str, str]] = {}
    for other in METHODS:
        if other == method or not results_path(config, other).exists():
            continue
        for family, held in load(config, other)["banks"].items():
            found.setdefault(family, {}).update({item["name"]: item["answer"] for item in held["items"]})
    return found

def resolve_lens(config: str, method: str, path: Optional[str]) -> Optional[JacobianLens]:
    """The lens a method reads through, downloaded on first use; None for the logit lens"""
    if method == "logit-lens":
        return None
    if path:
        return JacobianLens.load(path)
    if config not in LENSES[method]:
        raise SystemExit(f"no published {method} for '{config}' (have {sorted(LENSES[method])}); pass --lens <file>")
    from huggingface_hub import hf_hub_download

    repo, filename = LENSES[method][config]
    return JacobianLens.load(hf_hub_download(repo, filename), source=f"hf://{repo}/{filename}")

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

def run(config: str, families: List[str], limit: int = 0, methods: Sequence[str] = ("logit-lens",),
        lens_path: Optional[str] = None) -> None:
    """Every method in turn on one loaded model, each its own MLflow run and results file

    The model is loaded once for all of them: on the 27B a load is four
    minutes and the one window in which the machine runs short of memory.
    """
    adapter = load_adapter(config)
    for method in methods:
        # each method logs to its own file, which its MLflow run uploads; stdout still gets everything
        set_log_file(log_path(config, method))
        with track(f"{method}-{config}", EXPERIMENT,
                   outputs=[results_path(config, method), log_path(config, method)],
                   params={"config": config, "method": method, "banks": ",".join(families), "limit": limit,
                           "top_k": 10},
                   tags={"script": "wsbench_baseline", "method": method}) as tracker:
            scored(adapter, config, families, limit, method, lens_path, tracker)

def scored(adapter, config: str, families: List[str], limit: int, method: str, lens_path: Optional[str],
           tracker: Tracker) -> None:
    state = load(config, method)
    lens = resolve_lens(config, method, lens_path)
    if lens is not None:
        lens.check(adapter)
        state["lens"] = {"source": lens.source, "n_prompts": lens.n_prompts, "layers": len(lens.layers),
                         "provenance": lens.provenance}
        tracker.log_params({"lens": lens.source, "lens_n_prompts": lens.n_prompts,
                            **{f"lens_{key}": value for key, value in lens.provenance.items()
                               if key in ("dataset_id", "target_layer", "skip_first", "config_json")}})
    answers = earlier_answers(config, method)
    tracker.log_params({"model": adapter.cfg.hf_name, "n_layers": adapter.cfg.n_layers,
                        "d_model": adapter.cfg.d_model, "batch_size": adapter.cfg.batch_size,
                        "dtype": adapter.cfg.dtype, "max_new_tokens": adapter.cfg.max_new_tokens,
                        "answers_reused": bool(answers)})
    banner(f"{method} on WorkspaceBench", {
        "model": f"{adapter.cfg.hf_name} ({adapter.cfg.n_layers} layers, d_model {adapter.cfg.d_model})",
        "lens": state.get("lens", {}).get("source", "none -- the unembedding alone"),
        "banks": len(families),
        "answers": f"reused from {len(answers)} banks already run" if answers else "generated",
        "batch": adapter.cfg.batch_size,
        "writing": results_path(config, method),
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
        scores = score_bank(adapter, items, family=family, top_k=10,
                            transport=lens.transport if lens is not None else None,
                            answers=answers.get(family))
        summary = report(scores)
        summary["seconds"] = round(time.perf_counter() - started, 1)
        state["banks"][family] = {"summary": summary, "gate": bank.gate,
                                  "items": [asdict(score) for score in scores]}
        save(state)
        send_bank(tracker, families.index(family), family, summary)

        log(f"{family}: {summary['items']} items in {duration(summary['seconds'])} · {gpu()}", indent=1)
        log(f"hit {summary['hit_rate']:.1%} · clean {summary['clean_hit_rate']:.1%} · "
            f"leaked {summary['leak_rate']:.1%} · first hit at layer {summary['median_first_hit_layer']} · "
            f"degeneracy {summary['degeneracy']:.1%}", indent=2)

    overall(state)
    if state["banks"]:
        tracker.log(len(families), {f"all.{key}": value for key, value in totals(state).items()})
        tracker.flush()
    log(f"-> {results_path(config, method)}")

def backfill(config: str, method: str = "logit-lens") -> None:
    """Send a results file written before this script tracked, without loading the model

    The run is tagged `backfill` so it is not mistaken for a fresh
    measurement: its provenance is the commit that *sent* it, not the one
    that computed it, and its timestamps are today's.
    """
    state = load(config, method)
    if not state["banks"]:
        raise SystemExit(f"no results at {results_path(config, method)} to backfill")
    families = sorted(state["banks"])
    # no `outputs`: those upload what changed during the block, and a backfilled file changed long before it
    with track(f"{method}-{config}", EXPERIMENT,
               params={"config": config, "method": method, "banks": ",".join(families), "top_k": 10},
               tags={"script": "wsbench_baseline", "method": method, "backfill": "true"}) as tracker:
        for step, family in enumerate(families):
            send_bank(tracker, step, family, state["banks"][family]["summary"])
        tracker.log(len(families), {f"all.{key}": value for key, value in totals(state).items()})
        tracker.flush()
        for path in (results_path(config, method), log_path(config, method)):
            tracker.log_artifact(path, name=str(path))
        log(f"backfilled {len(families)} banks of {method} on {config}: "
            f"{tracker.url or tracker.failure or 'tracking off'}")
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
    """<config> [--method M] [--lens FILE] [limit] [bank ...]; --backfill sends a results file instead"""
    args = sys.argv[1:]
    config = args.pop(0) if args else "qwen3.6-27b"

    def option(flag: str) -> Optional[str]:
        if flag not in args:
            return None
        index = args.index(flag)
        value = args[index + 1] if index + 1 < len(args) else None
        del args[index:index + 2]
        return value

    methods = (option("--method") or "logit-lens").split(",")
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise SystemExit(f"--method takes a comma-separated list of {METHODS}, got {unknown}")
    lens_path = option("--lens")
    if lens_path and len(methods) > 1:
        raise SystemExit("--lens names one file, so it goes with one --method")
    if "--backfill" in args:
        for method in methods:
            backfill(config, method)
        return
    limit = int(args.pop(0)) if args and args[0].isdigit() else 0
    run(config, args or list(SINGLE_TOKEN_BANKS), limit=limit, methods=methods, lens_path=lens_path)

if __name__ == "__main__":
    main()
