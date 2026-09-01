"""Phase 1b deliverable 4b: is the single-head ranking a measurement or a coin flip?

The sweep scored 288 single heads on sentences 0-99 and produced a
distribution centred on zero -- mean -0.037, stdev 0.214, 45.8% of heads
"helping" the model when ablated. A ranking of that is a ranking of draws
unless the draws reproduce, and nothing in the sweep asks whether they do.
The whole-layer groups already say they do not compose: ablating all 32 heads
of layer 28 *improves* BLEU by 0.51 while ablating its best single head costs
0.51, and a superset cannot help where its subset hurts.

So this is the split-half check. The same heads are re-scored on sentences
100-199 -- disjoint from the first half, same protocol, same counterfactual
means -- and the two halves are correlated. Under a real effect the halves
agree and the strong heads stay strong. Under noise the extremes regress to
the mean, which is the one thing a single sample can never show you about
itself.

Heads are picked from both tails and the middle rather than from the top
alone. Taking only the winners measures regression without a comparison; the
middle group is what says whether the spread is wider than chance anywhere at
all.

The reading:

- halves correlate, tails hold   -> the ranking is a measurement, greedy may use it;
- halves do not correlate        -> the head half of the sweep is noise, and the
  deliverable-4 finding is "not localized to single heads at this granularity",
  which phase1b_random_control's docstring already names as a finding rather
  than a bug;
- correlated but tiny            -> real and too small to build a set from, which
  is a different decision than either of the above.

Budgeted and resumable like its siblings: each head is flushed as it is
scored, --budget exits cleanly mid-run, re-running resumes.

A common pipe could be: sweep_ranking | pick_heads | ablate | bleu_of | correlate

Run: uv run python -m scripts.phase1b_splithalf qwen3-8b run 10 1800
     uv run python -m scripts.phase1b_splithalf qwen3-8b report
"""

import json
import random
import time
from dataclasses import replace
from pathlib import Path

from scripts.observe import Budget, banner, degeneracy, duration, gpu, log, preview, set_log_file, step
from scripts.phase1b_ablation import (
    EVAL_SENTENCES,
    GENERATION_BATCH,
    HEAD_SENTENCES,
    SWEEP,
    ablate,
    bleu_of,
    capture_means,
    eval_data,
    reference_prompts,
    translate,
)
from src.model.adapter import load_adapter

PROGRESS = Path("results/phase1b-splithalf-progress.json")
REPORT = Path("results/phase1b-splithalf.json")

# The second half: disjoint from the sentences the sweep scored heads on, and
# the same size, so a dBLEU from here is in the units of a dBLEU from there.
SECOND_HALF = slice(HEAD_SENTENCES, EVAL_SENTENCES)

def load_progress() -> dict:
    return json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {"heads": {}}

def save_progress(state: dict) -> None:
    PROGRESS.parent.mkdir(exist_ok=True)
    PROGRESS.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")

def pick_heads(per_group: int) -> dict:
    """Both tails and the middle of the first-half ranking

    The tails are where regression to the mean would show; the middle is the
    control that says whether the spread exceeds chance anywhere.
    """
    if not SWEEP.exists():
        raise SystemExit(f"{SWEEP} does not exist; run the sweep and its assemble stage first")
    ranked = [row for row in json.loads(SWEEP.read_text())["ranked"]
              if row["component"].startswith("head:")]
    ranked.sort(key=lambda row: -row["dbleu"])
    middle = ranked[len(ranked) // 2 - per_group // 2:][:per_group] if per_group else []
    # a fixed seed so a resumed run picks the same heads as the run it resumes
    random.Random(0).shuffle(middle)
    return {
        "top": ranked[:per_group],
        "bottom": ranked[-per_group:],
        "middle": middle,
    }

def stage_run(config: str, per_group: int, budget: float) -> None:
    set_log_file("results/phase1b-splithalf.log")
    allowance = Budget(budget)
    state = load_progress()
    groups = pick_heads(per_group)
    plan = [(group, row) for group, rows in groups.items() for row in rows
            if row["component"] not in state["heads"]]
    banner("phase1b split-half reproducibility", {
        "config": config,
        "first half": f"sentences 0-{HEAD_SENTENCES - 1} (the sweep's own scores)",
        "second half": f"sentences {HEAD_SENTENCES}-{EVAL_SENTENCES - 1} (scored here)",
        "heads": f"{len(plan)} to do, {len(state['heads'])} already scored "
                 f"({per_group} per group x 3 groups)",
        "budget": duration(budget),
        "progress": str(PROGRESS),
    })
    if not plan:
        log("every selected head already scored -- run the report stage")
        return
    adapter = load_adapter(config)
    adapter.cfg = replace(adapter.cfg, batch_size=GENERATION_BATCH)
    wmt, prompts, references = eval_data()
    prompts, references = prompts[SECOND_HALF], references[SECOND_HALF]
    log(f"model loaded: {adapter.cfg.n_layers} layers x {adapter.cfg.n_heads} heads · {gpu()}")
    means = capture_means(adapter, reference_prompts(wmt))

    if "baseline" not in state:
        start = time.time()
        with step(f"baseline: {len(prompts)} sentences (second half), un-ablated"):
            hypotheses = translate(adapter, prompts, label="baseline")
        state["baseline"] = {"bleu": bleu_of(hypotheses, references),
                             "degeneracy": degeneracy(hypotheses),
                             "seconds": round(time.time() - start, 1)}
        save_progress(state)
        log(f"second-half baseline BLEU {state['baseline']['bleu']}")
        preview(hypotheses, "baseline")
    base = state["baseline"]["bleu"]

    for index, (group, row) in enumerate(plan):
        cid = row["component"]
        if not allowance.fits(state["baseline"]["seconds"]):
            log(f"stopping cleanly: {allowance.state()} -- re-run the same command to resume")
            return
        log(f"[{index + 1}/{len(plan)}] {cid} ({group}) · first-half dBLEU {row['dbleu']} · "
            f"{allowance.state()}")
        start = time.time()
        with ablate(adapter, means, [cid]):
            hypotheses = translate(adapter, prompts, label=cid)
        bleu = bleu_of(hypotheses, references)
        state["heads"][cid] = {
            "group": group,
            "first_half_dbleu": row["dbleu"],
            "second_half_bleu": bleu,
            "second_half_dbleu": round(base - bleu, 2),
            "degeneracy": degeneracy(hypotheses),
            "seconds": round(time.time() - start, 1),
        }
        save_progress(state)
        log(f"{cid}: first half {row['dbleu']:+.2f} · second half "
            f"{state['heads'][cid]['second_half_dbleu']:+.2f}", indent=1)
    stage_report()

def stage_report() -> None:
    """The correlation, and whether the tails held"""
    from scipy import stats

    state = load_progress()
    rows = [{"component": cid, **record} for cid, record in state.get("heads", {}).items()]
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} heads scored; run the run stage first")
    first = [row["first_half_dbleu"] for row in rows]
    second = [row["second_half_dbleu"] for row in rows]
    pearson = stats.pearsonr(first, second)
    spearman = stats.spearmanr(first, second)
    tops = [row for row in rows if row["group"] == "top"]
    held = sum(1 for row in tops if row["second_half_dbleu"] > 0)

    REPORT.write_text(json.dumps({
        "protocol": "the same single heads mean-ablated and scored on two disjoint 100-sentence halves of "
                    "the WMT shortlist, same counterfactual means and same protocol. A ranking that does "
                    "not reproduce across halves is a ranking of noise.",
        "first_half": f"sentences 0-{HEAD_SENTENCES - 1}, from {SWEEP}",
        "second_half": f"sentences {HEAD_SENTENCES}-{EVAL_SENTENCES - 1}, scored here",
        "second_half_baseline_bleu": state["baseline"]["bleu"],
        "heads_scored": len(rows),
        "pearson_r": round(float(pearson.statistic), 4),
        "pearson_p": round(float(pearson.pvalue), 5),
        "spearman_rho": round(float(spearman.statistic), 4),
        "spearman_p": round(float(spearman.pvalue), 5),
        "top_group_kept_sign": f"{held}/{len(tops)}",
        "verdict": ("the halves agree: the head ranking is a measurement"
                    if spearman.pvalue < 0.05 and spearman.statistic > 0 else
                    "the halves do not agree: the single-head ranking does not reproduce, and the "
                    "deliverable-4 finding is that the task is not localized to single heads at this "
                    "granularity"),
        "heads": sorted(rows, key=lambda row: -row["first_half_dbleu"]),
    }, indent=2, ensure_ascii=False) + "\n")

    log(f"split-half over {len(rows)} heads: pearson r={pearson.statistic:.3f} (p={pearson.pvalue:.4f}) · "
        f"spearman rho={spearman.statistic:.3f} (p={spearman.pvalue:.4f})")
    log(f"top group kept its sign on the second half: {held}/{len(tops)}", indent=1)
    for row in sorted(rows, key=lambda row: -row["first_half_dbleu"]):
        log(f"{row['component']:<14} {row['group']:<7} first {row['first_half_dbleu']:+.2f}  "
            f"second {row['second_half_dbleu']:+.2f}", indent=1)
    log(json.loads(REPORT.read_text())["verdict"])
    log(f"-> {REPORT}")

def main() -> None:
    import sys

    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b"
    stage = sys.argv[2] if len(sys.argv) > 2 else "run"
    if stage == "run":
        stage_run(config, int(sys.argv[3]) if len(sys.argv) > 3 else 10,
                  float(sys.argv[4]) if len(sys.argv) > 4 else 1800.0)
    elif stage == "report":
        stage_report()
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are run, report")

if __name__ == "__main__":
    main()
