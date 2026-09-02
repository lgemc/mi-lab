"""Phase 1b deliverable 4b: is the single-head ranking a measurement or a coin flip?

The sweep scored every single head on the first half of the shortlist and
produced a distribution centred on zero, with nearly half the heads
"helping" the model when ablated. A ranking of that is a ranking of draws
unless the draws reproduce, and nothing in the sweep asks whether they do.
The whole-layer groups already say they do not compose: ablating all of a
layer's heads can *improve* BLEU while ablating its best single head costs
the same amount, and a superset cannot help where its subset hurts.

So this is the split-half check. The same heads are re-scored on the second
half -- disjoint from the first, same protocol, same counterfactual means --
and the two halves are correlated (`methods.quality.agreement`). Under a real
effect the halves agree and the strong heads stay strong. Under noise the
extremes regress to the mean, which is the one thing a single sample can
never show you about itself.

Heads are picked from both tails and the middle rather than from the top
alone. Taking only the winners measures regression without a comparison; the
middle group is what says whether the spread is wider than chance anywhere at
all.

The reading:

- halves correlate, tails hold   -> the ranking is a measurement, greedy may use it;
- halves do not correlate        -> the head half of the sweep is noise, and the
  deliverable-4 finding is "not localized to single heads at this granularity";
- correlated but tiny            -> real and too small to build a set from, which
  is a different decision than either of the above.

Budgeted and resumable like its siblings: each head is flushed as it is
scored, --budget exits cleanly mid-run, re-running resumes.

Run: uv run python -m scripts.phase1b_splithalf qwen3-8b run 10 1800
     uv run python -m scripts.phase1b_splithalf qwen3-8b report
"""

import json
import random
import sys
from typing import Any, Dict, List

from src.experiment import translation_study as study
from src.methods.quality import agreement
from src.telemetry.observe import Budget, banner, duration, log, set_log_file
from src.telemetry.results import guard, load_state, save_state

PROGRESS = study.artifact("splithalf_progress")
REPORT = study.artifact("splithalf")
SWEEP = study.artifact("sweep")

def load_progress() -> Dict[str, Any]:
    return load_state(PROGRESS, {"heads": {}})

def save_progress(state: Dict[str, Any]) -> None:
    save_state(PROGRESS, state)

def pick_heads(per_group: int) -> Dict[str, List[Dict[str, Any]]]:
    """Both tails and the middle of the first-half ranking

    The tails are where regression to the mean would show; the middle is the
    control that says whether the spread exceeds chance anywhere.
    """
    if not SWEEP.exists():
        raise SystemExit(f"{SWEEP} does not exist; run the sweep and its assemble stage first")
    ranked = [row for row in load_state(SWEEP)["ranked"] if row["component"].startswith("head:")]
    ranked.sort(key=lambda row: -row["dbleu"])
    middle = ranked[len(ranked) // 2 - per_group // 2:][:per_group] if per_group else []
    random.Random(0).shuffle(middle)
    return {"top": ranked[:per_group], "bottom": ranked[-per_group:], "middle": middle}

def stage_run(config: str, per_group: int, budget: float) -> None:
    set_log_file(study.log_path("splithalf"))
    allowance = Budget(budget)
    state = load_progress()
    head, full = study.head_sentences(), study.eval_sentences()
    groups = pick_heads(per_group)
    plan = [(group, row) for group, rows in groups.items() for row in rows
            if row["component"] not in state["heads"]]
    banner("phase1b split-half reproducibility", {
        "config": config,
        "first half": f"sentences 0-{head - 1} (the sweep's own scores)",
        "second half": f"sentences {head}-{full - 1} (scored here)",
        "heads": f"{len(plan)} to do, {len(state['heads'])} already scored "
                 f"({per_group} per group x 3 groups)",
        "budget": duration(budget),
        "progress": str(PROGRESS),
    })
    if not plan:
        log("every selected head already scored -- run the report stage")
        return
    adapter, corpus, means = study.setup(config)
    second = slice(head, full)
    prompts, references = corpus.part(second)

    if "baseline" not in state:
        record = study.score(adapter, prompts, references, label="baseline")
        state["baseline"] = {key: record[key] for key in ("bleu", "degeneracy", "seconds")}
        save_progress(state)
        log(f"second-half baseline BLEU {state['baseline']['bleu']}")
    base = state["baseline"]["bleu"]

    for index, (group, row) in enumerate(plan):
        cid = row["component"]
        estimate = state["baseline"]["seconds"]
        if not allowance.fits(estimate):
            log(allowance.stop_line(estimate, "next head"))
            log("re-run the same command to resume")
            return
        log(f"[{index + 1}/{len(plan)}] {cid} ({group}) · first-half dBLEU {row['dbleu']} · "
            f"{allowance.state()}")
        record = study.score_set(adapter, means, corpus, [cid], label=cid, sentences=second)
        state["heads"][cid] = {
            "group": group,
            "first_half_dbleu": row["dbleu"],
            "second_half_bleu": record["bleu"],
            "second_half_dbleu": round(base - record["bleu"], 2),
            "degeneracy": record["degeneracy"],
            "seconds": record["seconds"],
        }
        save_progress(state)
        log(f"{cid}: first half {row['dbleu']:+.2f} · second half "
            f"{state['heads'][cid]['second_half_dbleu']:+.2f}", indent=1)
    stage_report()

def stage_report() -> None:
    """The correlation, and whether the tails held"""
    state = load_progress()
    head, full = study.head_sentences(), study.eval_sentences()
    rows = [{"component": cid, **record} for cid, record in state.get("heads", {}).items()]
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} heads scored; run the run stage first")
    stats = agreement([row["first_half_dbleu"] for row in rows], [row["second_half_dbleu"] for row in rows])
    tops = [row for row in rows if row["group"] == "top"]
    held = sum(1 for row in tops if row["second_half_dbleu"] > 0)

    REPORT.write_text(json.dumps({
        "protocol": "the same single heads mean-ablated and scored on two disjoint halves of the WMT "
                    "shortlist, same counterfactual means and same protocol. A ranking that does not "
                    "reproduce across halves is a ranking of noise.",
        "first_half": f"sentences 0-{head - 1}, from {SWEEP}",
        "second_half": f"sentences {head}-{full - 1}, scored here",
        "second_half_baseline_bleu": state["baseline"]["bleu"],
        "heads_scored": len(rows),
        "pearson_r": stats["pearson_r"],
        "pearson_p": stats["pearson_p"],
        "spearman_rho": stats["spearman_rho"],
        "spearman_p": stats["spearman_p"],
        "top_group_kept_sign": f"{held}/{len(tops)}",
        "verdict": ("the halves agree: the head ranking is a measurement" if stats["agrees"] else
                    "the halves do not agree: the single-head ranking does not reproduce, and the "
                    "deliverable-4 finding is that the task is not localized to single heads at this "
                    "granularity"),
        "heads": sorted(rows, key=lambda row: -row["first_half_dbleu"]),
    }, indent=2, ensure_ascii=False) + "\n")

    log(f"split-half over {len(rows)} heads: pearson r={stats['pearson_r']:.3f} (p={stats['pearson_p']:.4f}) · "
        f"spearman rho={stats['spearman_rho']:.3f} (p={stats['spearman_p']:.4f})")
    log(f"top group kept its sign on the second half: {held}/{len(tops)}", indent=1)
    for row in sorted(rows, key=lambda row: -row["first_half_dbleu"]):
        log(f"{row['component']:<14} {row['group']:<7} first {row['first_half_dbleu']:+.2f}  "
            f"second {row['second_half_dbleu']:+.2f}", indent=1)
    log(json.loads(REPORT.read_text())["verdict"])
    log(f"-> {REPORT}")

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    stage = sys.argv[2] if len(sys.argv) > 2 else "run"
    guard(config)
    if stage == "run":
        stage_run(config, int(sys.argv[3]) if len(sys.argv) > 3 else 10,
                  float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0)
    elif stage == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are run, report")

if __name__ == "__main__":
    main()
