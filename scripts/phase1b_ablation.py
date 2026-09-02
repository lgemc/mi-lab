"""Phase 1b deliverable 4: the mean-ablation sweep over the candidate components.

Mean-ablation knockout (Wang et al. 2211.00593 3, which is also the
validation step Zhang et al. 2502.11806 4.3 use -- their *discovery* method is
subspace-intervened path patching, which this sweep does not implement).
Each candidate component (an attention head or an MLP block in the band
`components.CANDIDATE_BAND` names) is replaced by its mean activation over the
counterfactual reference distribution, the ES->EN translation is regenerated
on the WMT shortlist, and the drop in corpus BLEU against the un-ablated
baseline is the component's score.

Everything this used to do itself now lives in `src/`: the means and the
hooks in `methods.knockout`, BLEU and the paired bootstrap in
`methods.quality`, the component vocabulary in `methods.components`, the
corpus, the baseline and the file names in `experiment.translation_study`.
What is left here is the protocol -- which components, on how many sentences,
in what order, written where -- and the resumable loop around it: every
finished component is flushed to the progress file and a wall-clock budget
makes each invocation exit cleanly, so the sweep runs as a chain of
foreground calls with nothing orphanable in the background.

Layer-level components (a whole layer's heads, one MLP) are scored on the full
set; single heads on half -- a solo head rarely moves corpus BLEU by more than
noise, so the halved pass buys double the coverage and the greedy stage
re-verifies everything it uses at the full count.

Stages: sweep (default) walks a component group; assemble writes the ranked
sweep file; significance is the paired bootstrap over the stored generations;
comet adds dCOMET for the top solo components.

Run: uv run python -m scripts.phase1b_ablation qwen3-8b sweep layers
     uv run python -m scripts.phase1b_ablation qwen3-8b sweep heads-upper
     uv run python -m scripts.phase1b_ablation qwen3-8b assemble
     uv run python -m scripts.phase1b_ablation qwen3-8b significance
     uv run python -m scripts.phase1b_ablation qwen3-8b comet
"""

import json
import sys
import time
from typing import Any, Dict

from src.experiment import translation_study as study
from src.methods import components as comp
from src.methods.knockout import Means, ablate, preview, translate
from src.methods.quality import BOOTSTRAP_RESAMPLES, SIGNIFICANCE_ALPHA, Comet, bleu, paired_significance
from src.telemetry.observe import Budget, banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, load_state, save_state

PROGRESS = study.artifact("ablation_progress")
SWEEP = study.artifact("sweep")
LOG = study.log_path("ablation")

# The group names the pipelines were written with, before the band was a
# depth fraction; they name the 8B's layers and mean the band halves.
GROUP_ALIASES = {"heads32-35": "heads-upper", "heads27-31": "heads-lower"}

def load_progress() -> Dict[str, Any]:
    return load_state(PROGRESS, {"components": {}})

def save_progress(state: Dict[str, Any]) -> None:
    save_state(PROGRESS, state)

def migrate_baseline(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Read a baseline written before the counts were part of the record

    The old keys named the numbers they held -- bleu_200, bleu_100 -- which was
    honest while the counts were constants and became a lie the moment they were
    not. Everything already on disk was scored at 200/100, so that is what the
    old names are read as.
    """
    if "bleu_eval" not in baseline and "bleu_200" in baseline:
        baseline = {**baseline, "bleu_eval": baseline["bleu_200"],
                    "bleu_head": baseline["bleu_100"], "sentences": {"eval": 200, "head": 100}}
    return baseline

def ensure_baseline(state: Dict[str, Any], adapter, corpus: study.Corpus) -> None:
    """The un-ablated score at both counts, once; a baseline at other counts is refused, not reused"""
    eval_count, head_count = study.eval_sentences(), study.head_sentences()
    if "baseline" in state:
        state["baseline"] = migrate_baseline(state["baseline"])
        counts = state["baseline"].get("sentences", {})
        if counts and (counts["eval"], counts["head"]) != (eval_count, head_count):
            raise SystemExit(
                f"the cached baseline was scored on {counts['eval']}/{counts['head']} sentences and this "
                f"run wants {eval_count}/{head_count}. Every component score in this file is a "
                f"difference against that baseline, so mixing the two would compare drops measured on "
                f"different corpora. Point MI_LAB_RESULTS at a fresh directory for the new protocol."
            )
        log(f"baseline already scored: BLEU {state['baseline']['bleu_eval']} (cached)")
        return
    with step(f"baseline: {len(corpus)} sentences, un-ablated") as facts:
        record = study.score(adapter, corpus.prompts, corpus.references, label="baseline")
        hypotheses = record["hypotheses"]
        state["baseline"] = {
            "bleu_eval": record["bleu"],
            "bleu_head": bleu(hypotheses[:head_count], corpus.references[:head_count]),
            "sentences": {"eval": eval_count, "head": head_count},
            "seconds": record["seconds"],
            "hypotheses": hypotheses,
        }
        facts["BLEU"] = f"{record['bleu']} at {eval_count} / {state['baseline']['bleu_head']} at {head_count}"
    save_progress(state)

def stage_sweep(config: str, group: str, budget: float) -> None:
    set_log_file(LOG)
    group = GROUP_ALIASES.get(group, group)
    allowance = Budget(budget)
    state = load_progress()
    banner("phase1b ablation sweep", {
        "config": config,
        "group": group,
        "already scored": len(state["components"]),
        "budget": duration(budget),
        "reference": "counterfactual prompt form (eval prompt minus translation logic)",
        "progress": str(PROGRESS),
    })
    adapter, corpus, means = study.setup(config)
    layers = means.layers
    whole = comp.plan(group, layers, adapter.cfg.n_heads)
    plan = [cid for cid in whole if cid not in state["components"]]
    log(f"candidate band {comp.CANDIDATE_BAND[0]:.2f}-{comp.CANDIDATE_BAND[1]:.2f} of depth on "
        f"{adapter.cfg.n_layers} layers -> {layers[0]}-{layers[-1]} ({len(layers)} layers, "
        f"{adapter.cfg.n_heads} heads each) · {len(plan)} to do of {len(whole)} in group")
    if not plan:
        log(f"group '{group}' already swept -- nothing to do")
        return
    ensure_baseline(state, adapter, corpus)
    eval_count, head_count = study.eval_sentences(), study.head_sentences()

    seen = []
    for index, cid in enumerate(plan):
        left = len(plan) - index
        estimate = sum(seen) / len(seen) if seen else state["baseline"]["seconds"]
        if not allowance.fits(estimate):
            log(allowance.stop_line(estimate, "next component"))
            log(f"{left} of {len(plan)} left in group '{group}' -- re-run the same command to resume")
            return
        kind, _, _ = comp.parse(cid)
        whole_layer = kind in ("mlp", "heads")
        count = eval_count if whole_layer else head_count
        base = state["baseline"]["bleu_eval"] if whole_layer else state["baseline"]["bleu_head"]
        started = time.time()
        log(f"[{index + 1}/{len(plan)}] {cid} · {count} sentences · {allowance.state()} · "
            f"eta for group {duration(estimate * left)}")
        prompts, references = corpus.part(slice(0, count))
        with ablate(adapter, means, [cid]):
            hypotheses = translate(adapter, prompts, label=cid)
        seen.append(time.time() - started)
        score = bleu(hypotheses, references)
        state["components"][cid] = {
            "bleu": score,
            "dbleu": round(base - score, 2),
            "sentences": count,
            "seconds": round(seen[-1], 1),
            # single heads keep no generations: the split-half check showed
            # their ranking does not reproduce, so nothing downstream tests them
            "hypotheses": hypotheses if whole_layer else [],
        }
        save_progress(state)
        log(f"{cid}: BLEU {score} (d {round(base - score, 2)}) in {duration(seen[-1])} · {gpu()}")
        preview(hypotheses, cid)
    log(f"group '{group}' done: {len(plan)} components in {duration(allowance.spent)}")

def stage_assemble(command: str) -> None:
    state = load_progress()
    ranked = sorted(
        ({"component": cid, **{key: value for key, value in record.items() if key != "hypotheses"}}
         for cid, record in state["components"].items()),
        key=lambda record: -record["dbleu"],
    )
    means = study.artifact("means")
    SWEEP.write_text(json.dumps({
        "protocol": "mean ablation (Wang et al. 2211.00593 knockout) per component, mean taken over the "
                    "counterfactual prompt distribution -- eval prompt with the translation logic removed; "
                    "greedy translation of the WMT shortlist (few-shot, 64 new tokens); "
                    "score = baseline corpus BLEU minus ablated corpus BLEU. Layer-level components on "
                    f"{study.eval_sentences()} sentences, single heads on {study.head_sentences()}.",
        "baseline": {key: value for key, value in state["baseline"].items() if key != "hypotheses"},
        "counterfactual_mean_tokens": Means.load(means).tokens if means.exists() else None,
        "components_scored": len(ranked),
        "wall_seconds_total": round(sum(record["seconds"] for record in ranked) + state["baseline"]["seconds"], 1),
        "ranked": ranked,
        "comet_top": state.get("comet_top"),
        "command": command,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(ranked)} components -> {SWEEP}; top 5: "
          + ", ".join(f"{r['component']} d{r['dbleu']}" for r in ranked[:5]))

def stage_significance(config: str) -> None:
    """Paired bootstrap of every component with stored generations against the baseline

    No GPU. Single heads are absent by construction -- the sweep stores no
    hypotheses for them -- and that absence is the finding rather than a gap
    to fill: phase1b_splithalf scored 30 of them on a disjoint half and the
    ranking did not reproduce, so there is nothing there to test.
    """
    set_log_file(LOG)
    state = load_progress()
    if "baseline" not in state:
        raise SystemExit(f"{PROGRESS} has no baseline; run the sweep first")
    count = study.eval_sentences()
    corpus = study.Corpus.load()
    testable = {cid: record["hypotheses"] for cid, record in state["components"].items()
                if len(record.get("hypotheses") or []) >= count}
    if not testable:
        raise SystemExit("no component stored a full set of generations, so nothing can be tested")
    banner("phase1b ablation significance", {
        "config": config,
        "testable": f"{len(testable)} of {len(state['components'])} components have stored generations",
        "not testable": f"{len(state['components']) - len(testable)} single heads (no generations stored; "
                        "phase1b_splithalf found their ranking does not reproduce)",
        "test": f"paired bootstrap, {BOOTSTRAP_RESAMPLES} resamples, {count} sentences",
        "alpha": SIGNIFICANCE_ALPHA,
    })
    names = sorted(testable, key=lambda cid: -state["components"][cid]["dbleu"])
    state["significance"] = paired_significance(
        state["baseline"]["hypotheses"], {cid: testable[cid] for cid in names}, corpus.references)
    save_progress(state)
    table = state["significance"]["components"]
    log(f"{'component':<12} {'dBLEU':>7} {'p':>8} {'q(FDR)':>8}   verdict")
    for cid in names:
        record = table[cid]
        verdict = ("significant (FDR)" if record["significant_fdr"]
                   else "raw only" if record["significant"] else "not significant")
        log(f"{cid:<12} {record['dbleu']:>7.2f} {record['p']!s:>8} {record['q']!s:>8}   {verdict}")
    passing = [cid for cid in names if table[cid]["significant"]]
    fdr = [cid for cid in names if table[cid]["significant_fdr"]]
    log(f"raw p < {SIGNIFICANCE_ALPHA}: {len(passing)} -> {passing}")
    log(f"FDR q < {SIGNIFICANCE_ALPHA}: {len(fdr)} -> {fdr}")
    if not fdr:
        log("!! nothing survives correction for the number of components screened. The raw-p survivors are "
            "the best candidates this sweep has, and they are not an established result -- treat them as a "
            "hypothesis to test on more data, not as a discovered circuit.")

def stage_comet(config: str, top: int) -> None:
    """dCOMET for the strongest solo components: regenerate at the full count where needed, score against baseline"""
    state = load_progress()
    ranked = sorted(state["components"].items(), key=lambda item: -item[1]["dbleu"])[:top]
    adapter, corpus, means = study.setup(config)
    comet = Comet()
    count = study.eval_sentences()

    results = {"baseline": comet.score(corpus.sources, state["baseline"]["hypotheses"], corpus.references)}
    for cid, record in ranked:
        hypotheses = record["hypotheses"]
        if len(hypotheses) < count:
            with ablate(adapter, means, [cid]):
                hypotheses = translate(adapter, corpus.prompts, label=cid)
        results[cid] = comet.score(corpus.sources, hypotheses, corpus.references)
        print(cid, results[cid], flush=True)
    state["comet_top"] = {
        "scores": results,
        "dcomet": {cid: round(results["baseline"] - value, 4) for cid, value in results.items() if cid != "baseline"},
    }
    save_progress(state)

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    stage = sys.argv[2] if len(sys.argv) > 2 else "sweep"
    guard(config)
    if stage == "sweep":
        group = sys.argv[3] if len(sys.argv) > 3 else "layers"
        budget = float(sys.argv[4]) if len(sys.argv) > 4 else 420.0
        stage_sweep(config, group, budget)
    elif stage == "assemble":
        stage_assemble(f"uv run python -m scripts.phase1b_ablation {config} sweep <group> (chained), then assemble")
    elif stage == "significance":
        stage_significance(config)
    elif stage == "comet":
        stage_comet(config, top=int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are sweep, assemble, significance, comet")

if __name__ == "__main__":
    main()
