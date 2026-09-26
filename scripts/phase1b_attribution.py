"""Phase 1b, revised: attribution patching as the discovery method, checked against the knockout sweep.

`docs/methodology.tex` retired greedy knockout as a *discovery* method after
the 2026 literature sweep -- MIB (2504.13151) finds attribution and
mask-optimisation methods best for circuit localisation, and greedy knockout
is on neither list -- and pre-registered EAP-IG as the replacement with
knockout kept for validation. This is that replacement, and the reason it can
be believed here rather than cited is that the validation already ran: the
306 components in `phase1b-ablation-sweep.json` cost 7,575 seconds of
generation and are the ranking this one has to reproduce in two passes.

The stages are the three choices around the primitive, which is the whole
content of the method:

  rank        score the lattice once, with a given metric, integration and
              granularity. Writes `phase1b-attribution.json`.
  agreement   the ranking against the sweep's dBLEU: Spearman over every
              shared component, overlap at the top, and what each cost.
  metrics     the same lattice under `target` and under `kl`, and how much
              the circuit changes -- Zhang & Nanda's claim, measured here.
  steps       plain EAP against the integrated estimate, and which of the two
              tracks the real ablation better. This is the saturation
              question: a first-order term is exactly wrong where the metric
              is flat.
  edges       the same pass at edge granularity, plus the split that says
              how much of a source's attribution goes to one reader. A node
              score cannot answer that and it is what is lost when a method
              reports per-component numbers.

Run: uv run python -m scripts.phase1b_attribution qwen3-8b rank
     uv run python -m scripts.phase1b_attribution qwen3-8b agreement
     uv run python -m scripts.phase1b_attribution qwen3-8b metrics
     uv run python -m scripts.phase1b_attribution qwen3-8b steps
     uv run python -m scripts.phase1b_attribution gpt2-small edges --scope model
"""

import sys
import time
from typing import Any, Dict, List, Optional

from src.experiment import translation_study as study
from src.methods import attribution as attr
from src.telemetry.observe import banner, duration, log, set_log_file, step
from src.telemetry.results import guard, load_state, merge_section, save_state
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

REPORT = study.artifact("attribution")
AGREEMENT = study.artifact("attribution_agreement")
LOG = study.log_path("attribution")


def parse(argv: List[str]) -> Dict[str, Any]:
    """config, stage and the flags the stages share, so every stage reads them the same way"""
    positional = [word for word in argv if not word.startswith("--")]
    flags = {}
    for index, word in enumerate(argv):
        if word.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else "1"
            flags[word[2:]] = following if not following.startswith("--") else "1"
    return {
        "config": positional[0] if positional else study.DEFAULT_CONFIG,
        "stage": positional[1] if len(positional) > 1 else "rank",
        "scope": flags.get("scope", "candidate"),
        "metric": flags.get("metric", "target"),
        "against": flags.get("against", "self"),
        "steps": int(flags.get("steps", 1)),
        "sentences": int(flags["sentences"]) if "sentences" in flags else None,
        "batch": int(flags.get("batch", attr.DEFAULT_BATCH)),
    }


def prepare(options: Dict[str, Any]):
    """(adapter, layers, spans): the model, the base being scored, and what it is scored on

    The means are loaded through the study's own `setup`, so an attribution
    run ablates toward exactly the values the knockout sweep ablated toward.
    Scoring against a different counterfactual than the measurement it is
    compared with would make the comparison a fact about the two
    counterfactuals.
    """
    adapter, corpus, means = study.setup(options["config"], options["scope"], size=None)
    layers = study.scope_layers(adapter.cfg, options["scope"])
    baseline = None
    if options["against"] == "self":
        state = load_state(study.artifact("ablation_progress"), {})
        baseline = state.get("baseline")
        if baseline is None:
            raise study.StudyError(
                "no baseline generations on disk to score against; run scripts.phase1b_ablation "
                "first, or pass --against gold to score against the WMT references"
            )
    with step("building spans") as facts:
        spans = study.attribution_spans(adapter, corpus, options["against"], baseline,
                                        size=options["sentences"])
        facts["spans"] = f"{len(spans)} sentences against {options['against']}"
    return adapter, means, layers, spans


def run(options: Dict[str, Any], metric: Optional[str] = None, steps: Optional[int] = None,
        granularity: str = "node", prepared=None) -> attr.Attribution:
    metric = metric or options["metric"]
    steps = steps if steps is not None else options["steps"]
    adapter, means, layers, spans = prepared if prepared is not None else prepare(options)
    label = f"{metric} steps={steps} {granularity}"
    with step(f"attribution: {label}") as facts:
        found = attr.attribute(adapter, means, spans, layers, metric_name=metric, steps=steps,
                               granularity=granularity, batch_size=options["batch"], label=label)
        facts["units"] = len(found.scores)
        facts["passes"] = found.passes
        facts["seconds"] = round(found.seconds, 1)
    return found


def against_knockout(found: attr.Attribution, n_heads: int) -> Dict[str, Any]:
    """The estimate against both slices of the sweep: everything, and what the sweep could resolve

    One function because three stages ask the same question and a second copy
    of "which slice, with which vocabulary" is a second chance to compare an
    estimate against a different measurement than the one it is reported
    beside.
    """
    grouped = attr.with_layer_groups(found, n_heads) if n_heads else found
    reports = {}
    for name, measured, estimate in (
        ("all", study.sweep_measured(), found),
        ("well_measured", study.sweep_measured(sentences=study.eval_sentences()), grouped),
    ):
        if measured:
            reports[name] = attr.agreement(estimate, measured, at=(5, 10, 20))
    return reports


def report_against_knockout(label: str, found: attr.Attribution, n_heads: int) -> Dict[str, Any]:
    reports = against_knockout(found, n_heads)
    for name, report in reports.items():
        log(f"{label:<12} vs knockout ({name}): spearman {report['spearman']:+.3f} over "
            f"{report['components']} components, overlap {report['overlap']}", indent=1)
    return reports


def stage_rank(options: Dict[str, Any]) -> None:
    adapter, means, layers, spans = prepare(options)
    found = run(options, prepared=(adapter, means, layers, spans))
    # the head count travels with the file: `agreement` sums a layer's heads
    # into its group and cannot ask a model that is no longer loaded
    record = found.to_dict()
    record["notes"] = {**record["notes"], "n_heads": adapter.cfg.n_heads}
    # The ranking in the vocabulary a set is grown in, positives only. It is
    # written and not acted on: `phase1b_greedy` grows its set down a ranking
    # *gated on the paired bootstrap*, because ranking single heads by point
    # estimate was the bug that gate was added for, and an attribution score
    # has no such gate -- it has no spread to test, being one number per
    # component rather than a sample. Feeding this list to the greedy walk is
    # therefore a decision to drop that protection, and it should be made on
    # purpose and written down, not made by a default here.
    record["proposed_components"] = attr.as_components(found)
    save_state(REPORT, record)
    log(f"{len(found.scores)} units scored in {found.passes} passes -> {REPORT}")
    for unit, score in found.ranked(10):
        log(f"{unit:<14} {score:+.4f}", indent=1)
    log(f"{len(record['proposed_components'])} components scored positive; the greedy walk's own "
        "ranking stays gated on the paired bootstrap and is not replaced by this file")


def stage_agreement(options: Dict[str, Any]) -> None:
    """The estimate against the measurement, on the part of the measurement that was measured well

    Twice, and the pair is the report. `all` is every component the sweep
    holds, which is dominated by 288 single heads scored on half the corpus
    and is mostly the sweep's own noise. `well_measured` is the components the
    sweep scored on the full set, expressed on the same vocabulary by summing
    each layer's heads into its `heads:L` group. A method that looks useless
    on the first and works on the second has not been rescued by the second:
    it means the estimate can rank what the sweep could resolve and nothing is
    known about the rest, which is a different and smaller claim.
    """
    if not REPORT.exists():
        raise study.StudyError(f"no attribution at {REPORT}; run the rank stage first")
    stored = load_state(REPORT)
    found = attr.Attribution.from_dict(stored)
    sweep_seconds = load_state(study.artifact("sweep")).get("wall_seconds_total")
    reports = against_knockout(found, int(stored.get("notes", {}).get("n_heads") or 0))
    for name, report in reports.items():
        report["knockout_seconds"] = sweep_seconds
        report["speedup"] = round(sweep_seconds / found.seconds, 1) if sweep_seconds and found.seconds else None
        merge_section(AGREEMENT, f"against_knockout_{name}", report)
        log(f"{name}: spearman {report['spearman']:+.3f} over {report['components']} components; "
            f"overlap {report['overlap']}")
        log("estimate top: " + ", ".join(report["estimate_top"][:8]), indent=1)
        log("measured top: " + ", ".join(report["measured_top"][:8]), indent=1)
    if sweep_seconds and found.seconds:
        log(f"{duration(found.seconds)} of attribution against {duration(sweep_seconds)} of knockout "
            f"({round(sweep_seconds / found.seconds, 1)}x)")


def stage_metrics(options: Dict[str, Any]) -> None:
    """The same base under both metrics: how much of a circuit is the metric's doing

    `kl` cannot be run at one step -- its gradient on the unablated model is
    zero -- so both sides are integrated here. That is the comparison being
    made: the metric changed and nothing else did.
    """
    prepared = prepare(options)
    steps = max(2, options["steps"])
    target = run(options, metric="target", steps=steps, prepared=prepared)
    kl = run(options, metric="kl", steps=steps, prepared=prepared)
    report = attr.disagreement(target, kl)
    merge_section(AGREEMENT, "metric_axis", report)
    log(f"target vs kl: spearman {report['spearman']:+.3f}, top-20 overlap {report['overlap_top20']:.2f}")
    n_heads = prepared[0].cfg.n_heads
    try:
        for name, found in (("target", target), ("kl", kl)):
            merge_section(AGREEMENT, f"metric_{name}_against_knockout",
                          report_against_knockout(name, found, n_heads))
    except study.StudyError as error:
        log(f"no knockout sweep to check either metric against ({error})", indent=1)


def stage_steps(options: Dict[str, Any]) -> None:
    """Plain EAP against the integrated estimate: the saturation question, as a number"""
    prepared = prepare(options)
    first = run(options, metric="target", steps=1, prepared=prepared)
    integrated = run(options, metric="target", steps=max(2, attr.DEFAULT_STEPS), prepared=prepared)
    report = attr.disagreement(first, integrated)
    merge_section(AGREEMENT, "integration_axis", report)
    log(f"eap vs integrated: spearman {report['spearman']:+.3f}, top-20 overlap {report['overlap_top20']:.2f}")
    n_heads = prepared[0].cfg.n_heads
    try:
        for name, found in (("eap", first), ("eap_ig_path", integrated)):
            merge_section(AGREEMENT, f"steps_{name}_against_knockout",
                          report_against_knockout(name, found, n_heads))
    except study.StudyError as error:
        log(f"no knockout sweep to check either estimator against ({error})", indent=1)


def stage_edges(options: Dict[str, Any]) -> None:
    """The same pass at edge granularity, and what the node summary of it cannot say"""
    found = run(options, granularity="edge")
    nodes = found.to_nodes()
    report = {"edges": found.to_dict(), "split": attr.split(found), "nodes": nodes.to_dict()}
    save_state(study.result("phase1b-attribution-edges.json"), report)
    log(f"{len(found.scores)} edges over {len(nodes.scores)} sources")
    for (source, destination), score in found.ranked(10):
        log(f"{source:<14} -> {destination:<10} {score:+.4f}", indent=1)
    log("how much of a source's attribution goes to one reader:")
    for row in report["split"]["rows"][:8]:
        log(f"{row['source']:<14} {row['destinations']} readers, "
            f"largest {row['largest']} at {row['largest_share']:.0%}", indent=1)


STAGES = {"rank": stage_rank, "agreement": stage_agreement, "metrics": stage_metrics,
          "steps": stage_steps, "edges": stage_edges}


def main(argv: List[str]) -> int:
    options = parse(argv)
    if options["stage"] not in STAGES:
        log(f"unknown stage '{options['stage']}'; stages are {', '.join(STAGES)}")
        return 2
    set_log_file(LOG)
    guard(options["config"])
    banner(f"phase 1b attribution: {options['stage']}", {
        "config": options["config"],
        "scope": options["scope"],
        "metric": options["metric"],
        "steps": options["steps"],
        "against": options["against"],
        "sentences": options["sentences"] or study.DEFAULT_ATTRIBUTION_SENTENCES,
        "report": REPORT,
    })
    started = time.time()
    STAGES[options["stage"]](options)
    log(f"stage '{options['stage']}' done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-translation-study", outputs=[results_root()]))
