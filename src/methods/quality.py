"""How a set of translations is scored, and when a difference between two sets is a finding.

The knockout study produces hypotheses -- one English sentence per Spanish
source, per ablated component -- and everything it claims is a comparison of
scores over those. Three scorers and two judgements live here so that every
script compares the same way:

- `bleu` is corpus BLEU through sacrebleu, the number every table reports.
- `comet` is COMET-22, the learned metric, loaded once and scored many times
  because the checkpoint is heavier than the numbers it produces.
- `paired_significance` is the paired bootstrap of every system against the
  baseline on the *same* sentences, with FDR correction over the number of
  systems screened, because eighteen tests at alpha .05 buy a false positive
  for free.
- `agreement` is whether two rankings of the same components on disjoint
  data agree, which is the split-half test: a ranking that does not
  reproduce is a ranking of noise.
- `survival_frontier` is the largest random-ablation share at which the
  model was still emitting language, from degeneracy rows, and the rule for
  reading it is the one that a lucky seed must not be allowed to bend.

Every number is rounded where it is produced, so a JSON diff between two runs
shows measurement changes and not float noise.

A common pipe could be: translate | bleu | paired_significance | agreement
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.metrics import MetricError, benjamini_hochberg

SIGNIFICANCE_ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 1000
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_BATCH = 16

class QualityError(MetricError):
    """A comparison that cannot be made of the hypotheses it was given"""

def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU, rounded to the two decimals a table shows"""
    import sacrebleu

    if len(hypotheses) != len(references):
        raise QualityError(f"{len(hypotheses)} hypotheses against {len(references)} references")
    return round(sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score, 2)

def bleu_signature(hypotheses: Sequence[str], references: Sequence[str]) -> str:
    """The score with its n-gram breakdown, then the settings that make it reproducible

    The breakdown (precisions, brevity penalty, lengths) is what a reader
    compares two tables on; the signature (tokenizer, casing, smoothing,
    version) is what somebody else's sacrebleu needs to land on the same number.
    """
    from sacrebleu.metrics import BLEU

    if len(hypotheses) != len(references):
        raise QualityError(f"{len(hypotheses)} hypotheses against {len(references)} references")
    metric = BLEU()
    score = metric.corpus_score(list(hypotheses), [list(references)])
    return f"{score.format()} | {metric.get_signature()}"

def chrf(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    """Corpus chrF, the character-level companion that does not reward a lucky n-gram"""
    import sacrebleu

    if len(hypotheses) != len(references):
        raise QualityError(f"{len(hypotheses)} hypotheses against {len(references)} references")
    return round(sacrebleu.corpus_chrf(list(hypotheses), [list(references)]).score, 2)

class Comet:
    """COMET-22 loaded once; `score` many times

    Held in a class rather than a function so the checkpoint load happens at
    a moment the caller chose, and so that a script that never scores never
    pays for the download.
    """

    def __init__(self, name: str = COMET_MODEL, gpus: int = 1):
        from comet import download_model, load_from_checkpoint

        self.model = load_from_checkpoint(download_model(name))
        self.gpus = gpus

    def score(self, sources: Sequence[str], hypotheses: Sequence[str], references: Sequence[str]) -> float:
        data = [{"src": source, "mt": hypothesis, "ref": reference}
                for source, hypothesis, reference in zip(sources, hypotheses, references, strict=True)]
        return round(float(self.model.predict(data, batch_size=COMET_BATCH, gpus=self.gpus).system_score), 4)

def paired_significance(baseline: Sequence[str], systems: Dict[str, Sequence[str]], references: Sequence[str],
                        resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = SIGNIFICANCE_ALPHA,
                        ) -> Dict[str, Any]:
    """Paired bootstrap of every system against the baseline on the same sentences, FDR-corrected

    No GPU: the generations are already on disk, and a significance test on
    them is arithmetic. Every system is truncated to the reference count so a
    system that generated more is compared on the sentences the others were.
    """
    from sacrebleu.metrics import BLEU
    from sacrebleu.significance import PairedTest

    count = len(references)
    if len(baseline) < count:
        raise QualityError(f"baseline has {len(baseline)} hypotheses for {count} references")
    short = [name for name, hypotheses in systems.items() if len(hypotheses) < count]
    if short:
        raise QualityError(f"systems with fewer than {count} hypotheses: {short}")
    if not systems:
        raise QualityError("no systems to test against the baseline")
    names = list(systems)
    outcome = PairedTest(
        [("baseline", list(baseline[:count])), *[(name, list(systems[name][:count])) for name in names]],
        {"bleu": BLEU()}, references=[list(references)], test_type="bs", n_samples=resamples,
    )()
    scores = outcome[1]["BLEU"]
    base = float(scores[0].score)
    raw = {name: float(entry.p_value) for name, entry in zip(names, scores[1:], strict=True)
           if entry.p_value is not None}
    qvalues = benjamini_hochberg(raw)
    per_system = {}
    for name, entry in zip(names, scores[1:], strict=True):
        per_system[name] = {
            "bleu": round(float(entry.score), 2),
            "dbleu": round(base - float(entry.score), 2),
            "p": round(raw[name], 5) if name in raw else None,
            "q": qvalues.get(name),
            "significant": bool(name in raw and raw[name] < alpha),
            "significant_fdr": bool(name in qvalues and qvalues[name] < alpha),
        }
    return {"alpha": alpha, "resamples": resamples, "sentences": count,
            "baseline_bleu": round(base, 2), "components": per_system}

def agreement(first: Sequence[float], second: Sequence[float], alpha: float = SIGNIFICANCE_ALPHA) -> Dict[str, Any]:
    """Pearson and Spearman between two scorings of the same items, and whether they agree

    `agrees` is a positive rank correlation at `alpha`: the sign matters
    because a ranking that reproduces *inverted* is not noise, it is a bug in
    one of the halves.
    """
    from scipy import stats

    if len(first) != len(second):
        raise QualityError(f"{len(first)} scores against {len(second)}")
    if len(first) < 4:
        raise QualityError(f"only {len(first)} items; a correlation over fewer than 4 is not a test")
    pearson = stats.pearsonr(first, second)
    spearman = stats.spearmanr(first, second)
    return {
        "pearson_r": round(float(pearson.statistic), 4),
        "pearson_p": round(float(pearson.pvalue), 5),
        "spearman_rho": round(float(spearman.statistic), 4),
        "spearman_p": round(float(spearman.pvalue), 5),
        "agrees": bool(spearman.pvalue < alpha and spearman.statistic > 0),
    }

def survival_frontier(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """(last share where every seed survived before the first break, the share that broke)

    A share survives only if *every* seed at it survived, and the frontier is
    the last share before the first break rather than the largest intact one
    anywhere. With one seed per level the two definitions agree; with several
    they do not, and `max` over intact rows would let a lucky draw at 9%
    outrank a break at 6% and report a frontier above a share already known
    to fail. Rows carry `share` and `degeneracy`.
    """
    by_share: Dict[float, List[Dict[str, Any]]] = {}
    for row in rows:
        by_share.setdefault(float(row["share"]), []).append(row)
    frontier = None
    broke_at = None
    for share in sorted(by_share):
        if all(row["degeneracy"] == 0.0 for row in by_share[share]):
            frontier = share
        else:
            broke_at = share
            break
    return frontier, broke_at
