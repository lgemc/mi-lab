"""Phase 1b deliverables 5-6: combo diagnostics, greedy set selection, COMET test.

Solo ablations in the sweep moved BLEU by at most ~1.6 points, so the set
question -- how much does the *candidate space* carry jointly -- is answered
first by three upper-bound combos (all candidate MLPs, all candidate heads,
both together = the full candidate set). The greedy stage then grows a set
from the solo ranking: at each step the highest-ranked unused component
joins, the union is re-ablated and re-scored on the full shortlist, and the
run stops when the marginal dBLEU stays under the saturation margin for
three consecutive additions. This is rank-ordered greedy (one pass per
step); full greedy re-evaluation of every remaining candidate each round
would cost O(n^2) generation passes and is recorded as out of budget.

The ranking is gated on the paired bootstrap `phase1b_ablation significance`
writes, and growth is bounded by the survival frontier `phase1b_random_control
frontier` measures: dBLEU keeps rising long after the model has stopped
producing language, so a set grown past the frontier is a measurement of
destruction, not a stronger circuit. The saturation rule, the pre-registered
ceiling and the COMET threshold are `experiment.translation_study` constants.

Stages: combo | greedy | comet (baseline + candidate set + truncations).

Run: uv run python -m scripts.phase1b_greedy qwen3-8b combo
     uv run python -m scripts.phase1b_greedy qwen3-8b greedy 420
     uv run python -m scripts.phase1b_greedy qwen3-8b comet
"""

import json
import sys
import time
from typing import Any, Dict, List

from scripts.phase1b_ablation import PROGRESS, load_progress, migrate_baseline, save_progress
from src.experiment import translation_study as study
from src.methods.quality import Comet
from src.telemetry.observe import Budget, banner, duration, log, set_log_file
from src.telemetry.results import guard, load_state

CANDIDATE = study.artifact("candidate")
FRONTIER = study.artifact("frontier")
LOG = study.log_path("greedy")

def baseline_bleu(state: Dict[str, Any]) -> float:
    return migrate_baseline(state["baseline"])["bleu_eval"]

def run_combo(adapter, means, corpus: study.Corpus, components: List[str], label: str = "combo") -> Dict[str, Any]:
    return study.score_set(adapter, means, corpus, components, label=label, cost=study.cost_model())

def stage_combo(config: str) -> None:
    set_log_file(LOG)
    state = load_progress()
    state.setdefault("combos", {})
    banner("phase1b upper-bound combos", {
        "config": config,
        "done": f"{len(state['combos'])} already scored",
        "reference": "counterfactual prompt form (eval prompt minus translation logic)",
        "progress": str(PROGRESS),
    })
    adapter, corpus, means = study.setup(config)
    combos = study.combos(means.layers)
    log(f"combos: {', '.join(combos)}")
    for name, components in combos.items():
        if name in state["combos"]:
            log(f"{name} already scored: BLEU {state['combos'][name]['bleu']} (cached)")
            continue
        record = run_combo(adapter, means, corpus, components, label=name)
        state["combos"][name] = {
            "components": components,
            "bleu": record["bleu"],
            "dbleu": round(baseline_bleu(state) - record["bleu"], 2),
            "seconds": record["seconds"],
            "hypotheses": record["hypotheses"],
        }
        save_progress(state)
        log(f"{name}: BLEU {record['bleu']} · dBLEU {state['combos'][name]['dbleu']} · "
            f"{duration(record['seconds'])}")

def solo_ranking(state: Dict[str, Any]) -> List[str]:
    """Components that beat the baseline significantly, by solo dBLEU, strongest first

    Ranking by dBLEU alone was the original form and it ranks noise: single
    heads landed at a mean near zero with half of them "helping" the model
    when ablated, a split-half of 30 of them did not reproduce, and the
    whole-layer groups contradict their own members. Growing a set down that
    order is assembling coin flips, and the saturation margin cannot stop it
    because the noise floor is wider than the threshold.

    So the ranking is gated on the paired bootstrap. A component with no entry
    is not silently dropped as if it had failed: heads store no generations,
    so they were never testable, and that is a different statement from
    having been tested and found wanting. Both end in exclusion here and the
    caller's log says which one applied.
    """
    significance = state.get("significance")
    if not significance:
        raise SystemExit(
            "no significance test has been run, so every ranking available here is a ranking of dBLEU "
            "point estimates whose own spread is wider than the effects they order. Run "
            "`uv run python -m scripts.phase1b_ablation <config> significance` first -- it needs no GPU."
        )
    tested = significance["components"]
    passing = [cid for cid in state["components"] if tested.get(cid, {}).get("significant")]
    return sorted(passing, key=lambda cid: -state["components"][cid]["dbleu"])

def frontier_gate(greedy: Dict[str, Any], state: Dict[str, Any]) -> float | None:
    """The ceiling growth stops at, or a refusal when the frontier ran and found none

    `frontier_share()` returns None for two situations that are not the same:
    nobody has run the frontier, and the frontier ran and found that this
    model survives nothing it was asked to survive. The second is a result --
    the stronger one -- and it is refused unless overridden, in which case the
    artifact carries the override so nobody downstream reads the set as one
    grown inside a measurable budget.
    """
    ceiling = study.frontier_share()
    measured = study.frontier_measured()
    if ceiling is None and measured:
        record = load_state(FRONTIER)
        if not study.past_frontier_allowed():
            raise SystemExit(
                f"the frontier ran and found none: {record.get('first_broken_share')} of model MACs already "
                "breaks this model on at least one draw, and that is the smallest share probed. There is no "
                "ablation budget here inside which a knockout measures localization rather than destruction, "
                "so growing a set would produce a number with nothing behind it.\n\n"
                "That is a finding about the model, not a missing input. Probe smaller shares if you think "
                "the frontier is merely below the grid, or take the result: this model does not host a "
                "measurable circuit claim at this granularity.\n\n"
                f"{study.ENV_ALLOW_PAST_FRONTIER}=1 grows the set anyway and stamps the artifact with what it "
                "cost, for when the sequence has to run end to end regardless."
            )
        greedy["grown_without_frontier"] = {
            "first_broken_share": record.get("first_broken_share"),
            "note": f"{study.ENV_ALLOW_PAST_FRONTIER}=1 was set. No probed ablation share left this model "
                    "intact on every seed, so every number grown below is a measurement of a model "
                    "that has partly stopped emitting language. It is not evidence of localization.",
        }
        save_progress(state)
        log(f"!! {study.ENV_ALLOW_PAST_FRONTIER}=1: growing a set on a model with no survival frontier. "
            f"{record.get('first_broken_share')} of MACs already breaks it on at least one draw. "
            "Every number from here measures destruction; it is recorded in the artifact as such.")
    elif ceiling is None:
        log("!! no survival frontier measured, so nothing stops this growing into a broken model. "
            "Run `frontier` in phase1b_random_control first; continuing on marginal dBLEU alone.")
    else:
        log(f"growth stops at the measured survival frontier: {ceiling:.1%} of model MACs")
    return ceiling

def stage_greedy(config: str, budget: float) -> None:
    set_log_file(LOG)
    allowance = Budget(budget)
    state = load_progress()
    state.setdefault("greedy", {"chosen": [], "trajectory": []})
    greedy = state["greedy"]
    banner("phase1b greedy set growth", {
        "config": config,
        "resuming at": f"{len(greedy['chosen'])} components chosen, "
                       f"{len(greedy['trajectory'])} steps recorded",
        "saturation": f"marginal dBLEU < {study.SATURATION_MARGIN} for {study.SATURATION_RUNS} "
                      "consecutive additions",
        "budget": duration(budget),
        "progress": str(PROGRESS),
    })
    adapter, corpus, means = study.setup(config)
    baseline = baseline_bleu(state)
    cost = study.cost_model()

    ranking = solo_ranking(state)
    significance = state["significance"]
    untested = len(state["components"]) - len(significance["components"])
    fdr = [cid for cid, record in significance["components"].items() if record.get("significant_fdr")]
    log(f"ranking gated on the paired bootstrap: {len(ranking)} of {len(state['components'])} components "
        f"pass raw p < {significance['alpha']} ({untested} were never testable, no stored generations)")
    log(f"surviving FDR correction across the {len(significance['components'])} tested: "
        f"{fdr if fdr else 'none'}", indent=1)
    if not fdr:
        log("!! the set grown below rests on raw p-values that do not survive correction for the number "
            "of components screened. It is a hypothesis, not a discovered circuit.", indent=1)
    if not ranking:
        raise SystemExit("no component passed the significance gate; there is no set to grow")
    ceiling = frontier_gate(greedy, state)

    def marginals() -> List[float]:
        return [entry["marginal_dbleu"] for entry in greedy["trajectory"]]

    steps = []
    stopped_at_frontier = False
    while not study.saturated(marginals()):
        estimate = sum(steps) / len(steps) if steps else state["baseline"]["seconds"]
        if not allowance.fits(estimate):
            log(allowance.stop_line(estimate, "next step"))
            log(f"{len(greedy['chosen'])} components chosen so far -- re-run the same command to continue")
            return
        cid = study.next_candidate(ranking, greedy["chosen"])
        if cid is None:
            break
        trial = [*greedy["chosen"], cid]
        if ceiling is not None and cost.share(trial) > ceiling:
            log(f"stopping at the frontier: adding {cid} would cost {cost.share(trial):.2%} of model MACs, "
                f"past the {ceiling:.1%} the model was measured to survive. "
                f"{len(greedy['chosen'])} components chosen.")
            stopped_at_frontier = True
            break
        log(f"[step {len(greedy['trajectory']) + 1}] adding {cid} -> {len(trial)} components · "
            f"{allowance.state()}")
        started = time.time()
        record = run_combo(adapter, means, corpus, trial, label=f"greedy+{cid}")
        steps.append(time.time() - started)
        cumulative = round(baseline - record["bleu"], 2)
        previous = greedy["trajectory"][-1]["cumulative_dbleu"] if greedy["trajectory"] else 0.0
        greedy["chosen"] = trial
        greedy["trajectory"].append({
            "added": cid,
            "bleu": record["bleu"],
            "cumulative_dbleu": cumulative,
            "marginal_dbleu": round(cumulative - previous, 2),
        })
        save_progress(state)
        print(f"+{cid}: BLEU {record['bleu']} cumulative d {cumulative} "
              f"(marginal {round(cumulative - previous, 2)})", flush=True)
    if stopped_at_frontier:
        outcome = (f"greedy stopped at the survival frontier ({ceiling:.1%} of MACs) -- the set is bounded "
                   "by what the model survives, not by saturation")
    elif study.saturated(marginals()):
        outcome = "greedy saturated"
    else:
        outcome = "greedy exhausted the candidate ranking"
    print(outcome, flush=True)

def frontier_note(ceiling, measured: bool, candidate_share: float, interpretable) -> str:
    if ceiling is None and measured:
        return ("the frontier ran and found none: no probed share left this model intact on every "
                "seed, so there is no budget inside which this candidate's dCOMET measures "
                "localization rather than destruction, and the pre-registered pass above is not "
                "evidence of a circuit")
    if ceiling is None:
        return ("no frontier has been measured, so nothing here establishes that the ablated model "
                "was still a language model; run the frontier stage of phase1b_random_control")
    verdict = ("inside it, so its dCOMET is a translation loss" if interpretable else
               "outside it, so its dCOMET measures destruction rather than localization and the "
               "pre-registered pass above is not evidence of a circuit")
    return (f"the model was measured to survive ablation up to {ceiling:.1%} of MACs; this "
            f"candidate costs {candidate_share:.2%}, {verdict}")

def stage_comet(config: str) -> None:
    state = load_progress()
    greedy = state["greedy"]
    adapter, corpus, means = study.setup(config)
    cost = study.cost_model()
    comet = Comet()

    def score(hypotheses) -> float:
        return comet.score(corpus.sources, hypotheses, corpus.references)

    chosen = greedy["chosen"]
    full = state["combos"]["full_candidate_set"]
    is_full = set(chosen) == set(full["components"])
    evaluations: Dict[str, Dict[str, Any]] = {
        "baseline": {"bleu": baseline_bleu(state), "comet22": score(state["baseline"]["hypotheses"])},
        "candidate": {"components": chosen, "flops_share": round(cost.share(chosen), 4),
                      "bleu": full["bleu"] if is_full else None,
                      "comet22": score(full["hypotheses"]) if is_full else None},
    }
    if evaluations["candidate"]["comet22"] is None:
        record = run_combo(adapter, means, corpus, chosen, label="candidate")
        evaluations["candidate"].update(bleu=record["bleu"], comet22=score(record["hypotheses"]))
    # the smallest greedy prefixes: the pre-registered test asks for *a* set under
    # the ceiling, and the interesting one is the cheapest that still clears it
    for size in (1, 2, 4):
        prefix = chosen[:size]
        record = run_combo(adapter, means, corpus, prefix, label=f"greedy_prefix_{size}")
        evaluations[f"greedy_prefix_{size}"] = {
            "components": prefix, "flops_share": round(cost.share(prefix), 4),
            "bleu": record["bleu"], "comet22": score(record["hypotheses"]),
        }
        print(f"prefix {size}: BLEU {record['bleu']} COMET {evaluations[f'greedy_prefix_{size}']['comet22']}",
              flush=True)
    for name, record in evaluations.items():
        if name == "baseline":
            continue
        record["dcomet"] = round(evaluations["baseline"]["comet22"] - record["comet22"], 4)
        record["dbleu"] = round(evaluations["baseline"]["bleu"] - record["bleu"], 2)
    threshold = study.COMET_THRESHOLD
    passing = [name for name, record in evaluations.items()
               if name != "baseline" and record["dcomet"] >= threshold
               and record["flops_share"] <= study.PREREGISTERED_CEILING]
    smallest = min(passing, key=lambda name: evaluations[name]["flops_share"], default=None)

    # The pre-registered criterion is reported exactly as pre-registered, ceiling
    # and all -- rewriting a threshold after seeing the data is the thing
    # pre-registration exists to prevent. The frontier is reported *beside* it,
    # because a set can satisfy the criterion and still be uninterpretable: past
    # the frontier every set clears any threshold for the same reason an
    # unplugged model would. So `pass` keeps its pre-registered meaning and
    # `interpretable` says whether that pass is evidence.
    ceiling = study.frontier_share()
    measured = study.frontier_measured()
    candidate_share = evaluations["candidate"]["flops_share"]
    if ceiling is not None:
        interpretable = bool(candidate_share <= ceiling)
    elif measured:
        interpretable = False    # a set of any size is outside a budget that does not exist
    else:
        interpretable = None
    CANDIDATE.write_text(json.dumps({
        "protocol": "greedy rank-ordered set growth over the solo-ablation ranking, cumulative mean-ablation "
                    "re-scored on the full WMT shortlist each step; saturation = marginal dBLEU < "
                    f"{study.SATURATION_MARGIN} for {study.SATURATION_RUNS} consecutive additions. COMET-22 on "
                    "the same sentences for the pre-registered test.",
        "greedy_trajectory": greedy["trajectory"],
        "combos": {name: {key: value for key, value in record.items() if key != "hypotheses"}
                   for name, record in state.get("combos", {}).items()},
        "evaluations": evaluations,
        "pre_registered_test": {
            "criterion": f"component set <= {study.PREREGISTERED_CEILING:.0%} FLOPs whose ablation drops "
                         f"COMET-22 by >= {threshold}",
            "threshold_dcomet": threshold,
            "candidate_dcomet": evaluations["candidate"]["dcomet"],
            "candidate_flops_share": candidate_share,
            "pass": bool(evaluations["candidate"]["dcomet"] >= threshold
                         and candidate_share <= study.PREREGISTERED_CEILING),
            "smallest_passing_set": smallest,
            "smallest_passing_flops_share": evaluations[smallest]["flops_share"] if smallest else None,
        },
        "survival_frontier": {
            "share": ceiling,
            "source": str(FRONTIER),
            "candidate_flops_share": candidate_share,
            "interpretable": interpretable,
            "measured": measured,
            "note": frontier_note(ceiling, measured, candidate_share, interpretable),
        },
        "command": f"uv run python -m scripts.phase1b_greedy {config} combo|greedy|comet (chained)",
    }, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(json.loads(CANDIDATE.read_text())["pre_registered_test"], indent=2))
    if interpretable is False:
        where = (f"past the {ceiling:.1%} survival frontier" if ceiling is not None
                 else "and this model has no survival frontier at all")
        print(f"\n!! WARNING: the candidate costs {candidate_share:.2%} of model MACs, {where}. "
              "Every set this large scores a large dCOMET because the model has stopped emitting "
              "language, so the pre-registered pass above is not evidence of a circuit. "
              "Read the generations.", flush=True)
    elif interpretable is None:
        print("\n!! WARNING: no survival frontier measured; nothing establishes that the ablated model "
              "was still a language model.", flush=True)

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else study.DEFAULT_CONFIG
    stage = sys.argv[2] if len(sys.argv) > 2 else "combo"
    guard(config)
    if stage == "combo":
        stage_combo(config)
    elif stage == "greedy":
        stage_greedy(config, budget=float(sys.argv[3]) if len(sys.argv) > 3 else 420.0)
    elif stage == "comet":
        stage_comet(config)
    else:
        raise SystemExit(f"unknown stage '{stage}'; stages are combo, greedy, comet")

if __name__ == "__main__":
    main()
