"""Phase 1b deliverable 7: the random-component control arm at matched FLOPs.

The knockout in phase1b_ablation.py says how much BLEU a discovered set costs
when it is ablated. On its own that number is unfalsifiable: a model spreads a
task across enough components that an arbitrary set of the same size takes a
real bite out of it too, and a reference distribution that removes more than
the task takes the same bite out of *everything*. Zhang et al. (2502.11806)
report their knockouts only as Key Heads vs Random Heads and Key MLPs vs
Random MLPs for exactly this reason; without the second curve, "ablating these
nine MLPs drops BLEU by d" is a number with nothing to be larger than.

So this arm draws component sets that cost the same MACs as the discovered
one (`methods.cost.matched_draw`), ablates them toward the same
counterfactual means, and scores them the same way. Three readings come out:

- the discovered set clears the random band  -> the localization is doing work;
- the two coincide                            -> the damage is the ablation, not
  the circuit, and the reference distribution is the first suspect;
- the random band is itself huge              -> the task is not localized at
  this granularity, which is a finding rather than a bug.

Scope defaults to the whole stack rather than the candidate band. A control
drawn only from layers 1a already flagged asks "are these the right components
*within* the right layers"; drawn from the model it asks "is the localization
real at all", which is the question the arm exists to settle.

Every arm also reports a degeneracy score (`core.metrics.degeneracy`): the
fraction of hypotheses that have stopped being language. Corpus BLEU cannot
tell "translates badly" from "emits `a a a a a`", and the first arm to check
is the one at the largest ablation. The frontier stage turns that into a
curve: how much of the model can be mean-ablated before it stops being a
language model, which is the budget inside which any knockout is evidence.

Resumable and budgeted like its siblings: each seed is flushed as it finishes,
--budget makes an invocation exit cleanly mid-sweep, and re-running picks up
where it stopped.

Stages: run draws and scores the arms; frontier probes the shares; report and
frontier-report write the JSON files and print the verdicts.

Run: uv run python -m scripts.phase1b_random_control qwen3-8b frontier model 1 1800
     uv run python -m scripts.phase1b_random_control qwen3-8b run model all_candidate_mlps 5 1800
     uv run python -m scripts.phase1b_random_control qwen3-8b run model greedy 8 900
     uv run python -m scripts.phase1b_random_control qwen3-8b run candidate greedy 8 900
     uv run python -m scripts.phase1b_random_control qwen3-8b report
"""

import json
import sys
from typing import Any, Dict, List

from src.experiment import translation_study as study
from src.methods.cost import MATCH_TOLERANCE, matched_draw
from src.methods.quality import survival_frontier
from src.telemetry.observe import Budget, banner, duration, log, set_log_file
from src.telemetry.results import guard, load_state, save_state

PROGRESS = study.artifact("control_progress")
CONTROL = study.artifact("control")
FRONTIER = study.artifact("frontier")

def load_progress() -> Dict[str, Any]:
    return load_state(PROGRESS, {"arms": {}})

def save_progress(state: Dict[str, Any]) -> None:
    save_state(PROGRESS, state)

def target_set(target: str, layers: List[int]) -> List[str]:
    """The component set the control is matched to

    `greedy` is the end-state use: match the set the greedy stage settled on.
    The combo names are the *early* use, and the more important one -- they
    need no sweep to have run, so the control can be the first thing on the
    GPU rather than the last: `all_candidate_mlps` is the exact set that once
    collapsed the output to repeated tokens, and asking a matched random set
    the same question costs one pass and settles whether the collapse was the
    circuit or the ablation before any sweep budget is spent.
    """
    combos = study.combos(layers)
    if target in combos:
        return combos[target]
    if target != "greedy":
        raise SystemExit(f"unknown target '{target}'; targets are greedy and {sorted(combos)}")
    try:
        return study.candidate_components()
    except study.StudyError as error:
        raise SystemExit(
            f"{error}. Either run the sweep and the greedy stage first, or match a pre-registered combo "
            f"instead -- targets {sorted(combos)} need nothing to have run."
        ) from None

def draw(adapter, scope: str, share: float, seed: int) -> List[str]:
    chosen, error = matched_draw(study.pool(adapter.cfg, scope), study.cost_model(), share, seed)
    if error > MATCH_TOLERANCE:
        log(f"!! seed {seed}: the closest draw is {error:.2%} off the target share, "
            f"past the {MATCH_TOLERANCE:.0%} tolerance", indent=1)
    return chosen

def score_arm(adapter, means, corpus: study.Corpus, components: List[str], label: str) -> Dict[str, Any]:
    return study.score_set(adapter, means, corpus, components, label=label, cost=study.cost_model())

def stage_run(config: str, scope: str, target: str, seeds: int, budget: float) -> None:
    set_log_file(study.log_path("random-control"))
    allowance = Budget(budget)
    state = load_progress()
    key = f"{scope}/{target}"
    done = len([k for k in state["arms"].get(key, {}) if k.startswith("random_")])
    banner("phase1b random-component control", {
        "config": config,
        "scope": f"{scope} ({'whole stack' if scope == 'model' else 'candidate band'})",
        "target": target,
        "seeds": f"{seeds} requested, {done} already scored",
        "budget": duration(budget),
        "reference": "counterfactual prompt form (eval prompt minus translation logic)",
        "progress": str(PROGRESS),
    })
    # means must cover every layer the control may ablate, not just the candidate band
    adapter, corpus, means = study.setup(config, scope=scope)
    chosen = target_set(target, study.band(adapter.cfg))
    share = study.cost_model().share(chosen)
    log(f"target {target}: {len(chosen)} components · {share:.4f} of model MACs")
    study.ensure_baseline(state, adapter, corpus, save_progress)
    base = state["baseline"]["bleu"]

    arms = state["arms"].setdefault(key, {})
    if "discovered" not in arms:
        arms["discovered"] = score_arm(adapter, means, corpus, chosen, label=f"discovered/{target}")
        save_progress(state)
        found = round(base - arms["discovered"]["bleu"], 2)
        log(f"discovered ({target}): BLEU {arms['discovered']['bleu']} · dBLEU {found} · "
            f"degeneracy {arms['discovered']['degeneracy']}")
        if arms["discovered"]["degeneracy"]:
            log("!! degeneracy is non-zero on the discovered arm: the outputs are repeated tokens, not "
                "translation. This is a broken model, not an ablated one -- the reference distribution is "
                "the first suspect. Read the samples above before trusting any dBLEU from this run.")
    else:
        log(f"discovered ({target}) already scored: BLEU {arms['discovered']['bleu']} (cached)")
    found = round(base - arms["discovered"]["bleu"], 2)

    for seed in range(seeds):
        name = f"random_{seed}"
        if name in arms:
            continue
        estimate = arms["discovered"]["seconds"]
        if not allowance.fits(estimate):
            log(allowance.stop_line(estimate, "next seed"))
            log(f"{seeds - len([k for k in arms if k.startswith('random_')])} seeds left -- "
                "re-run the same command to resume")
            return
        log(f"[seed {seed + 1}/{seeds}] {allowance.state()}")
        arms[name] = score_arm(adapter, means, corpus, draw(adapter, scope, share, seed), label=name)
        arms[name]["match_error"] = round(abs(arms[name]["flops_share"] - share) / share, 4)
        save_progress(state)
        this = round(base - arms[name]["bleu"], 2)
        log(f"{name}: BLEU {arms[name]['bleu']} · dBLEU {this} · {arms[name]['n_components']} components · "
            f"match {arms[name]['match_error']:.2%} · degeneracy {arms[name]['degeneracy']}")
        band = [round(base - a["bleu"], 2) for k, a in arms.items() if k.startswith("random_")]
        verdict = "ahead" if found > max(band) else "NOT ahead"
        log(f"running verdict: discovered dBLEU {found} vs random band "
            f"{min(band)}..{max(band)} over {len(band)} seeds -- discovered is {verdict}", indent=1)
    log(f"scope '{scope}' target '{target}' done in {duration(allowance.spent)} -- "
        "run the report stage next")

def stage_frontier(config: str, scope: str, seeds: int, budget: float) -> None:
    """How much of the model can be mean-ablated before it stops being a language model

    Random components on purpose. The frontier wanted here is a property of
    the *model and the intervention*, not of any candidate set -- "what does
    ablating this much cost, whatever it is" -- so a discovered set that later
    beats this curve at the same share is a set doing something the average
    component does not. The shares are dense where the model was seen to
    break, because the interval between "intact" and "broken" is where the
    pre-registered combos sit.
    """
    set_log_file(study.log_path("survival-frontier"))
    allowance = Budget(budget)
    state = load_progress()
    levels = state.setdefault("frontier", {}).setdefault(scope, {})
    banner("phase1b survival frontier", {
        "config": config,
        "scope": scope,
        "shares": ", ".join(f"{share:.1%}" for share in study.FRONTIER_SHARES),
        "seeds per level": seeds,
        "done": f"{len(levels)} levels already scored",
        "budget": duration(budget),
    })
    adapter, corpus, means = study.setup(config, scope=scope)
    study.ensure_baseline(state, adapter, corpus, save_progress)

    for share in study.FRONTIER_SHARES:
        for seed in range(seeds):
            name = f"{share:.3f}/{seed}"
            if name in levels:
                log(f"{name} already scored: BLEU {levels[name]['bleu']} (cached)")
                continue
            estimate = state["baseline"]["seconds"]
            if not allowance.fits(estimate):
                log(allowance.stop_line(estimate, "next level"))
                log("re-run the same command to continue")
                return
            log(f"[share {share:.1%} seed {seed}] {allowance.state()}")
            levels[name] = score_arm(adapter, means, corpus, draw(adapter, scope, share, seed),
                                     label=f"share{share:.1%}s{seed}")
            save_progress(state)
            log(f"{name}: BLEU {levels[name]['bleu']} · degeneracy {levels[name]['degeneracy']} · "
                f"{levels[name]['n_components']} components")
    stage_frontier_report(scope)

def stage_frontier_report(scope: str) -> None:
    """The curve, and the largest share at which the model was still emitting language"""
    state = load_progress()
    levels = state.get("frontier", {}).get(scope, {})
    if not levels:
        raise SystemExit(f"no frontier levels scored for scope '{scope}' yet; run the frontier stage first")
    base = state["baseline"]["bleu"]
    rows = []
    for name in sorted(levels, key=lambda key: (float(key.split("/")[0]), key)):
        record = levels[name]
        record["degeneracy"] = study.rerun_degeneracy(record)
        rows.append({
            "share": float(name.split("/")[0]),
            "seed": int(name.split("/")[1]),
            "n_components": record["n_components"],
            "bleu": record["bleu"],
            "dbleu": round(base - record["bleu"], 2),
            "degeneracy": record["degeneracy"],
        })
    frontier, broke_at = survival_frontier(rows)
    seeds_per_share: Dict[str, int] = {}
    for row in rows:
        seeds_per_share[str(row["share"])] = seeds_per_share.get(str(row["share"]), 0) + 1
    FRONTIER.write_text(json.dumps({
        "protocol": "random matched-FLOPs mean ablation at increasing shares of model MACs, scored on the "
                    "WMT shortlist. The frontier is the largest share at which every generation was "
                    "still language. A knockout outside it measures destruction, not localization.",
        "scope": scope,
        "baseline_bleu": base,
        "survival_frontier_share": frontier,
        "first_broken_share": broke_at,
        "seeds_per_share": seeds_per_share,
        "note": ("no probed share left the model intact -- the frontier is below the smallest share tried"
                 if frontier is None else
                 f"ablating up to {frontier:.1%} of model MACs left every generation as language on every "
                 f"seed; circuit claims are only measurable at or below this"
                 + (f", and {broke_at:.1%} is the first share that broke" if broke_at is not None else
                    ", and no probed share above it broke -- the frontier is at or above the largest tried")),
        "curve": rows,
    }, indent=2) + "\n")
    log(f"survival frontier ({scope}): "
        + ("below the smallest share probed" if frontier is None else f"{frontier:.1%} of model MACs"))
    for row in rows:
        mark = "ok" if row["degeneracy"] == 0.0 else "BROKEN"
        log(f"{row['share']:>7.1%}  {row['n_components']:>4} comps  BLEU {row['bleu']:>6}  "
            f"dBLEU {row['dbleu']:>6}  degeneracy {row['degeneracy']:>5}  {mark}", indent=1)
    log(f"-> {FRONTIER}")

def empties(record: Dict[str, Any]) -> int:
    return sum(1 for h in record.get("hypotheses", []) if not h.strip())

def stage_report(command: str) -> None:
    state = load_progress()
    if "baseline" not in state or not state["arms"]:
        raise SystemExit(f"{PROGRESS} has no scored arms yet; run the run stage first")
    base = state["baseline"]["bleu"]

    scopes = {}
    for scope, arms in state["arms"].items():
        if "discovered" not in arms:
            continue
        # degeneracy is recomputed from the stored generations rather than read
        # back from the record: the generations are the evidence, the score is a
        # threshold over them, and a run scored under an older threshold must not
        # report a number this build would not produce
        for record in arms.values():
            record["degeneracy"] = study.rerun_degeneracy(record)
        arms["discovered"]["empty"] = empties(arms["discovered"])
        randoms = [record for name, record in arms.items() if name.startswith("random_")]
        if not randoms:
            continue
        drops = sorted(round(base - record["bleu"], 2) for record in randoms)
        found = round(base - arms["discovered"]["bleu"], 2)
        best_random = max(drops)
        scopes[scope] = {
            "discovered": {key: value for key, value in arms["discovered"].items() if key != "hypotheses"},
            "discovered_dbleu": found,
            "random_seeds": len(randoms),
            "random_dbleu": {"min": drops[0], "median": drops[len(drops) // 2], "max": best_random, "all": drops},
            "random_flops_shares": sorted({record["flops_share"] for record in randoms}),
            "random_degeneracy_max": max(record["degeneracy"] for record in randoms),
            "random_empty_max": max(empties(record) for record in randoms),
            "discovered_empty": arms["discovered"]["empty"],
            "random_match_error_max": max(record.get("match_error", 0.0) for record in randoms),
            "match_tolerance": MATCH_TOLERANCE,
            "separation": round(found - best_random, 2),
            "clears_random_band": bool(found > best_random),
        }

    CONTROL.write_text(json.dumps({
        "protocol": "random-component control at matched FLOPs (Zhang et al. 2502.11806 Fig. 2 shape). Each "
                    "random arm is a component set drawn without replacement from the scope until its MACs "
                    f"match the discovered set's within {MATCH_TOLERANCE:.0%}, ablated toward the same "
                    "counterfactual means and scored on the same WMT shortlist. A discovered set that "
                    "does not clear the random band is not evidence of a circuit.",
        "scopes": {
            "candidate": "atomic components drawn from the candidate band only: are these the right "
                         "components within the right layers",
            "model": "atomic components drawn from the whole stack: is the localization real at all",
        },
        "baseline_bleu": base,
        "baseline_degeneracy": state["baseline"]["degeneracy"],
        "degeneracy_note": "fraction of hypotheses that are repeated tokens rather than language; any arm "
                           "with a non-zero value here has its BLEU read as a broken-model number, not a "
                           "translation-quality one",
        "results": scopes,
        "command": command,
    }, indent=2, ensure_ascii=False) + "\n")

    for scope, record in scopes.items():
        verdict = "CLEARS the random band" if record["clears_random_band"] else "DOES NOT clear the random band"
        print(f"[{scope}] discovered dBLEU {record['discovered_dbleu']} vs random max "
              f"{record['random_dbleu']['max']} over {record['random_seeds']} seeds -> {verdict} "
              f"(separation {record['separation']})")
        if record["random_degeneracy_max"] or record["discovered"]["degeneracy"]:
            print(f"[{scope}] WARNING: degeneracy is non-zero "
                  f"(discovered {record['discovered']['degeneracy']}, random max "
                  f"{record['random_degeneracy_max']}); read the generations before quoting any of this")
    print(f"-> {CONTROL}")

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    stage = sys.argv[2] if len(sys.argv) > 2 else "run"
    guard(config)
    if stage == "run":
        scope = sys.argv[3] if len(sys.argv) > 3 else "model"
        target = sys.argv[4] if len(sys.argv) > 4 else "greedy"
        seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        budget = float(sys.argv[6]) if len(sys.argv) > 6 else 900.0
        stage_run(config, scope, target, seeds, budget)
    elif stage == "frontier":
        scope = sys.argv[3] if len(sys.argv) > 3 else "model"
        seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        budget = float(sys.argv[5]) if len(sys.argv) > 5 else 1800.0
        stage_frontier(config, scope, seeds, budget)
    elif stage == "frontier-report":
        stage_frontier_report(sys.argv[3] if len(sys.argv) > 3 else "model")
    elif stage == "report":
        stage_report(f"uv run python -m scripts.phase1b_random_control {config} run <scope> (chained), then report")
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are run, frontier, frontier-report, report")

if __name__ == "__main__":
    main()
