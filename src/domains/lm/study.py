"""The Spanish-to-English knockout study as one object: its corpus, its band, its files, its baseline.

Phase 1b is six scripts that all ask the same model the same question --
translate these sentences with this component knocked out -- and each one
used to carry its own copy of how the corpus is split, where the means are
cached, what the baseline record looks like and which JSON file the others
read. Two copies of a protocol drift, and a drift here is a comparison
between numbers that were measured on different sentences.

So every fact the scripts have to agree on is stated once:

- `Corpus` is the evaluation set: few-shot prompts over the first N pairs,
  their references, and the counterfactual prompts the means are averaged
  over. N comes from the environment so a pipeline can set it for every step
  at once, and the same env var is what a smoke run at 20 sentences uses.
- `ARTIFACTS` names every file a phase1b script writes, under the results
  root `telemetry.results` resolves, so a script reads what another wrote
  by key rather than by retyping a filename.
- `setup` is the one way to get a model, a corpus and its means together,
  and `score_set` is the one way to score a component set against it, so a
  BLEU in one file is the same BLEU in the next.
- The pre-registered numbers -- the frontier ceiling, the COMET threshold,
  the greedy saturation rule -- live here as constants because the point of
  pre-registering is that no script gets to choose them at run time.

A common pipe could be: setup | ensure_baseline | score_set | cost_model | frontier_share
"""

import os
import time
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...core.metrics import degeneracy
from ...methods.common import components as comp
from ...methods.knockout.ablate import Means, ablate, cached_means
from ...methods.knockout.cost import CostModel
from ...telemetry.observe import gpu, log, step
from ...telemetry.results import load_state, result, root
from .analysis.generate import preview, translate
from .analysis.quality import bleu
from .data.translation import EvalSplit, counterfactual_prompts, default_pairs_path, eval_split, load_pairs

ENV_EVAL_SENTENCES = "MI_LAB_EVAL_SENTENCES"
DEFAULT_EVAL_SENTENCES = 200
ENV_ALLOW_PAST_FRONTIER = "MI_LAB_ALLOW_PAST_FRONTIER"

CORPUS = "wmt-newstest2013-es-en-500"
GENERATION_BATCH = 100      # sentences per generate() call; the study's models fit this on the host
DEFAULT_CONFIG = "qwen3-8b"

# Pre-registered before the greedy run and not adjustable from a script:
# a candidate above this share of the model's MACs is not a circuit, and a
# candidate must cost the model at least this much COMET to count as one.
PREREGISTERED_CEILING = 0.25
COMET_THRESHOLD = 0.020

# The greedy walk stops when this many consecutive additions each buy less
# than this much BLEU: what is left is noise being ranked.
SATURATION_MARGIN = 0.3
SATURATION_RUNS = 3

# Random-ablation shares the survival frontier is probed at, dense where the
# model was seen to break.
FRONTIER_SHARES = (0.005, 0.01, 0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20)

MEANS_PREFIX = "phase1b-counterfactual-means"
ARTIFACTS = {
    "cost": "phase1b-flops-model.json",
    "means": f"{MEANS_PREFIX}.pt",
    "ablation_progress": "phase1b-ablation-progress.json",
    "sweep": "phase1b-ablation-sweep.json",
    "candidate": "phase1b-circuit-candidate.json",
    "control_progress": "phase1b-random-control-progress.json",
    "control": "phase1b-random-control.json",
    "frontier": "phase1b-survival-frontier.json",
    "splithalf_progress": "phase1b-splithalf-progress.json",
    "splithalf": "phase1b-splithalf.json",
    "weights": "phase1b-circuit-weights.pt",
    "manifest": "phase1b-circuit-manifest.json",
    "extraction": "phase1b-circuit-extraction.json",
    "attribution": "phase1b-attribution.json",
    "attribution_agreement": "phase1b-attribution-agreement.json",
}

# What a component's attribution score is teacher-forced against. `self` is
# the unablated model's own generations, which is what `dbleu` is a departure
# from; `gold` is the WMT reference, which the model may never have produced.
TARGETS = ("self", "gold")

# Sentences an attribution pass runs over. Far fewer than a knockout sweep
# needs, because the cost is passes-per-sentence and not passes-per-component:
# the whole lattice is scored in the same two passes, so the sample size is
# chosen for the variance of the estimate rather than for a budget.
DEFAULT_ATTRIBUTION_SENTENCES = 50

class StudyError(ValueError):
    """A study file that is missing or was measured under a different protocol, with the way out"""

def artifact(key: str) -> Path:
    """One of the study's files under the active results root, by key"""
    if key not in ARTIFACTS:
        raise StudyError(f"no study artifact named '{key}'; known: {sorted(ARTIFACTS)}")
    return result(ARTIFACTS[key])

def log_path(name: str) -> Path:
    return result(f"phase1b-{name}.log")

def eval_sentences() -> int:
    """How many sentences a score is over -- from the environment, so a pipeline sets it for every step"""
    return int(os.environ.get(ENV_EVAL_SENTENCES, DEFAULT_EVAL_SENTENCES))

def head_sentences() -> int:
    """The cheaper first half, used to screen many heads before the survivors get the full set"""
    return eval_sentences() // 2

@dataclass(frozen=True)
class Corpus:
    """The pairs file as the study uses it: scored prompts, references, and what the means average over"""
    pairs: Tuple[Tuple[str, str], ...]
    split: EvalSplit

    @classmethod
    def load(cls, size: Optional[int] = None, name: str = CORPUS) -> "Corpus":
        pairs = tuple(load_pairs(str(default_pairs_path(name))))
        return cls(pairs=pairs, split=eval_split(pairs, size=size if size is not None else eval_sentences()))

    @property
    def prompts(self) -> List[str]:
        return self.split.prompts

    @property
    def references(self) -> List[str]:
        return self.split.references

    @property
    def sources(self) -> List[str]:
        return self.split.sources

    @property
    def counterfactual(self) -> List[str]:
        # the same shots the eval prompts carry, so the mean strips the task and nothing else
        return counterfactual_prompts(self.pairs, shots=len(self.split.shots))

    def part(self, sentences: slice) -> Tuple[List[str], List[str]]:
        """(prompts, references) for a slice of the scored set, for half-set scoring"""
        return self.prompts[sentences], self.references[sentences]

    def __len__(self) -> int:
        return len(self.split)

def load_model(config: str, batch_size: int = GENERATION_BATCH):
    """The adapter with the study's generation batch, logged the way every script logs it"""
    from ...model.adapter import load_adapter  # torch stays out of the module import

    with step(f"loading {config}") as facts:
        adapter = load_adapter(config)
        adapter.cfg = replace(adapter.cfg, batch_size=batch_size)
        facts["model"] = f"{adapter.cfg.n_layers} layers x {adapter.cfg.n_heads} heads, batch {batch_size}"
        facts["gpu"] = gpu()
    return adapter

def band(cfg) -> List[int]:
    """The candidate band as this model's layer indices"""
    return comp.band(cfg)

SCOPES = ("candidate", "model")

def scope_layers(cfg, scope: str) -> List[int]:
    """The layers a scope names: the candidate band, or the whole stack"""
    if scope == "candidate":
        return band(cfg)
    if scope == "model":
        return list(range(cfg.n_layers))
    raise StudyError(f"unknown scope '{scope}'; scopes are {', '.join(SCOPES)}")

def pool(cfg, scope: str) -> List[str]:
    """Every atomic component a control may draw from within a scope

    Atomic: one MLP or one head. `heads:L` is a group of them and would make
    a draw coarser than the lattice it is matched on.
    """
    return comp.atomic_components(scope_layers(cfg, scope), cfg.n_heads)

def means_path(layers: Sequence[int], candidate: Sequence[int]) -> Path:
    """The band's means under the canonical name; any other layer set under a name that says which"""
    layers = list(layers)
    if layers == list(candidate):
        return artifact("means")
    return result(f"{MEANS_PREFIX}-L{min(layers)}-{max(layers)}.pt")

def study_means(adapter, corpus: Corpus, layers: Optional[Sequence[int]] = None) -> Means:
    """Counterfactual means over `layers` (the band by default), cached beside the other means files"""
    candidate = band(adapter.cfg)
    layers = list(layers) if layers is not None else candidate
    return cached_means(adapter, corpus.counterfactual, layers, means_path(layers, candidate),
                        siblings=root().glob(f"{MEANS_PREFIX}*.pt"))

def setup(config: str, scope: str | Sequence[int] = "candidate", size: Optional[int] = None):
    """(adapter, corpus, means): everything a knockout pass needs, loaded in the order that reports best

    `scope` is a scope name or an explicit layer list; the means cover exactly
    those layers, which is every layer a pass on them may ablate.
    """
    adapter = load_model(config)
    layers = scope_layers(adapter.cfg, scope) if isinstance(scope, str) else list(scope)
    with step("loading corpus") as facts:
        corpus = Corpus.load(size)
        facts["sentences"] = f"{len(corpus)} scored, {len(corpus.counterfactual)} counterfactual"
    means = study_means(adapter, corpus, layers)
    return adapter, corpus, means

def score(adapter, prompts: Sequence[str], references: Sequence[str], label: str) -> Dict[str, Any]:
    """Translate and score: {bleu, degeneracy, seconds, hypotheses}, with the generations previewed"""
    started = time.time()
    hypotheses = translate(adapter, prompts, label=label)
    broken = preview(hypotheses, label=label)
    return {"bleu": bleu(hypotheses, references), "degeneracy": round(broken, 4),
            "seconds": round(time.time() - started, 1), "hypotheses": hypotheses}

def ensure_baseline(state: Dict[str, Any], adapter, corpus: Corpus, save) -> Dict[str, Any]:
    """The unablated score on the full scored set, measured once and kept in `state["baseline"]`

    A baseline on disk that was measured over a different sentence count is
    refused rather than reused: every dBLEU in the study is a difference
    against it, and the difference between 200 and 497 sentences is not a
    component's doing.
    """
    baseline = state.get("baseline")
    if baseline is not None:
        found = len(baseline.get("hypotheses", []))
        if found != len(corpus):
            raise StudyError(
                f"the baseline on disk has {found} hypotheses but the corpus has {len(corpus)}; "
                f"set {ENV_EVAL_SENTENCES} to match or point MI_LAB_RESULTS at a fresh directory"
            )
        return baseline
    with step("baseline") as facts:
        baseline = score(adapter, corpus.prompts, corpus.references, label="baseline")
        facts["BLEU"] = baseline["bleu"]
    state["baseline"] = baseline
    save(state)
    return baseline

def score_set(adapter, means: Means, corpus: Corpus, components: Sequence[str], label: str,
              cost: Optional[CostModel] = None, sentences: slice = slice(None)) -> Dict[str, Any]:
    """One component set knocked out: {components, n_components, [flops_share], bleu, degeneracy, seconds, hypotheses}

    `sentences` scores a slice of the set -- the first half for the cheap head
    screen, the second half for the split-half check.
    """
    prompts, references = corpus.part(sentences)
    with step(label) as facts, ablate(adapter, means, components):
        record = score(adapter, prompts, references, label=label)
        facts["BLEU"] = record["bleu"]
        facts["degeneracy"] = f"{record['degeneracy']:.1%}"
    record = {"components": list(components), "n_components": len(components), **record}
    if cost is not None:
        record["flops_share"] = round(cost.share(list(components)), 4)
    return record

@cache
def cost_model() -> CostModel:
    """The FLOPs table the study divides by, from the file `phase1b_flops` wrote"""
    path = artifact("cost")
    if not path.exists():
        raise StudyError(f"no cost model at {path}; run scripts.phase1b_flops first")
    return CostModel.from_dict(load_state(path))

def attribution_spans(adapter, corpus: Corpus, against: str = "self",
                      baseline: Optional[Dict[str, Any]] = None, size: Optional[int] = None):
    """The corpus as teacher-forced spans: each eval prompt and the continuation being scored

    `against="self"` scores the likelihood of what the unablated model
    actually generated, which is the quantity `dbleu` is a difference in:
    phase 1b measures how far ablation moves the output, and the reference
    only enters through BLEU. `against="gold"` scores the WMT reference
    instead, which is a different question -- how far ablation moves the model
    from a translation it may never have produced -- and on a model whose
    baseline BLEU is in the thirties those are not the same components.

    A single space is the joiner because the few-shot form ends in `English:`
    and its worked examples continue `English: <sentence>` -- so the scored
    continuation has to be tokenized the way the shots above it were, or the
    model is being scored on a sequence it was never shown the form of.
    `teacher_forced` refuses the pair if that space ends up inside a merge
    with the prompt's last token rather than starting the continuation.
    """
    from ...model.passes import teacher_forced  # torch stays out of the module import

    if against not in TARGETS:
        raise StudyError(f"attribution is scored against {' or '.join(TARGETS)}, got '{against}'")
    size = size if size is not None else DEFAULT_ATTRIBUTION_SENTENCES
    prompts, references = corpus.prompts[:size], corpus.references[:size]
    if against == "gold":
        return teacher_forced(adapter, prompts, references, joiner=" ")
    if baseline is None:
        raise StudyError(
            "scoring against the model's own output needs the baseline record its generations are in; "
            "run scripts.phase1b_ablation first, or pass against='gold'"
        )
    hypotheses = list(baseline.get("hypotheses") or [])
    if len(hypotheses) < size:
        raise StudyError(
            f"the baseline has {len(hypotheses)} generations and {size} sentences were asked for; "
            f"lower the count or re-measure the baseline at {ENV_EVAL_SENTENCES}={size}"
        )
    return teacher_forced(adapter, prompts, hypotheses[:size], joiner=" ")

def sweep_measured(path: Optional[Path] = None, sentences: Optional[int] = None) -> Dict[str, float]:
    """The knockout sweep as component -> dBLEU: the ranking an attribution estimate is checked against

    This is the expensive number. Each entry is one mean ablation and one
    generation pass over the scored set, and the file the study already has
    holds 306 of them for 7,575 seconds of wall clock. An attribution pass
    that reproduces this ranking has reproduced two hours of GPU time in two
    passes, and one that does not has said something about the first-order
    approximation rather than about the model.

    `sentences` filters to the rows measured over that many sentences, and it
    matters more than it looks. The sweep scored layer-level components on the
    full set and single heads on half, and on half the corpus a solo head's
    dBLEU has a spread of ~0.2 around zero while only ten of 288 reach +-0.5.
    Those rows are a noise floor, not a ranking, and anything compared against
    all 306 at once is being compared mostly against them.
    """
    path = path if path is not None else artifact("sweep")
    if not path.exists():
        raise StudyError(f"no knockout sweep at {path}; run scripts.phase1b_ablation assemble first")
    rows = load_state(path).get("ranked") or []
    return {row["component"]: float(row["dbleu"]) for row in rows
            if "component" in row and "dbleu" in row
            and (sentences is None or row.get("sentences") == sentences)}

def combos(layers: Sequence[int]) -> Dict[str, List[str]]:
    """The three whole-band sets the candidate is compared against"""
    return {
        "all_candidate_mlps": comp.layer_components(layers),
        "all_candidate_heads": [comp.name("heads", layer) for layer in layers],
        "full_candidate_set": comp.layer_components(layers) + [comp.name("heads", layer) for layer in layers],
    }

def candidate_components() -> List[str]:
    """The circuit `phase1b_greedy` settled on, from its candidate file"""
    path = artifact("candidate")
    if not path.exists():
        raise StudyError(f"no candidate at {path}; run scripts.phase1b_greedy first")
    try:
        return list(load_state(path)["evaluations"]["candidate"]["components"])
    except KeyError as error:
        raise StudyError(f"{path} has no evaluations.candidate.components ({error})") from None

def frontier_share() -> Optional[float]:
    """The survival frontier, if `phase1b_random_control frontier-report` has written one"""
    path = artifact("frontier")
    if not path.exists():
        return None
    return load_state(path).get("survival_frontier_share")

def frontier_measured() -> bool:
    return artifact("frontier").exists()

def frontier_ceiling() -> Tuple[float, str]:
    """(the share a candidate may not exceed, where that number came from)"""
    measured = frontier_share()
    if measured is not None:
        return measured, "survival frontier"
    return PREREGISTERED_CEILING, "pre-registered ceiling"

def past_frontier_allowed() -> bool:
    return os.environ.get(ENV_ALLOW_PAST_FRONTIER) == "1"

def saturated(marginals: Sequence[float], margin: float = SATURATION_MARGIN, runs: int = SATURATION_RUNS) -> bool:
    """Whether the last `runs` additions each bought less than `margin` BLEU"""
    return len(marginals) >= runs and all(gain < margin for gain in marginals[-runs:])

def next_candidate(ranking: Sequence[str], chosen: Sequence[str]) -> Optional[str]:
    """The best-ranked component not yet chosen and not already implied by one that is"""
    for cid in ranking:
        if cid in chosen or comp.redundant(cid, chosen):
            continue
        return cid
    return None

def rerun_degeneracy(record: Dict[str, Any]) -> float:
    """The degeneracy of a record's hypotheses, recomputed so an old record gets the current definition"""
    hypotheses = record.get("hypotheses") or []
    return round(degeneracy(hypotheses), 4) if hypotheses else float(record.get("degeneracy", 0.0))

def describe(components: Sequence[str], limit: int = 6) -> str:
    """A component list short enough for a log line"""
    shown = ", ".join(components[:limit])
    return shown + (f", … ({len(components)} total)" if len(components) > limit else "")

def log_ranked(rows: Sequence[Dict[str, Any]], key: str = "dbleu", limit: int = 10, indent: int = 1) -> None:
    for row in rows[:limit]:
        log(f"{row.get('component', row.get('name', '?')):<14} {key}={row.get(key)}", indent=indent)
