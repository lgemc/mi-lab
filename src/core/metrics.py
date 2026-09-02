import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import torch

"""
Every experiment in this framework is supposed to end in a number, and these
are the numbers. AUC and accuracy say whether a method works; Cost says what
it costs to run, which is the half that decides whether it ever ships.

AUC is computed rank-wise with ties averaged rather than by thresholding a
grid, so it is exact and does not quietly depend on how the scores happen to
be scaled.

A circuit study measures something else and it is here for the same reason:
logit_difference is what "the model got it right" means when the answer is one
of two tokens, and recovery is how a patched run is read against the clean and
corrupted runs it sits between. Both are arithmetic on numbers a model already
produced, so both stay model-free.

Two more for the ablation studies, model-free for the same reason.
`degeneracy` is the fraction of generations that have stopped being language,
which a corpus BLEU cannot distinguish from translating badly and which turned
a retracted 32x threshold pass into `a a a a a`. `benjamini_hochberg` is the
correction for having screened many components to find one.

A common pipe could be: score | roc_auc | best_threshold
"""

class MetricError(ValueError):
    """Raised when a metric is undefined for the data it was given, rather than silently returning 0.5"""

def _as_tensors(scores: Sequence[float], labels: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten scores and labels to matching 1-D tensors, in double precision"""
    score_tensor = torch.as_tensor(scores, dtype=torch.float64).flatten()
    label_tensor = torch.as_tensor(labels, dtype=torch.int64).flatten()
    if score_tensor.numel() != label_tensor.numel():
        raise MetricError(f"{score_tensor.numel()} scores but {label_tensor.numel()} labels")
    if score_tensor.numel() == 0:
        raise MetricError("no examples to score")
    return score_tensor, label_tensor

def _average_ranks(scores: torch.Tensor) -> torch.Tensor:
    """Rank the scores from 1 upwards, giving tied scores their shared mean rank"""
    ordered, order = scores.sort()
    positions = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    _, inverse, counts = torch.unique_consecutive(ordered, return_inverse=True, return_counts=True)
    totals = torch.zeros(counts.numel(), dtype=torch.float64).index_add_(0, inverse, positions)
    ranks = torch.empty_like(scores)
    ranks[order] = (totals / counts)[inverse]
    return ranks

def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve: the chance a random positive outranks a random negative

    0.5 is the coin flip. A probe below 0.5 is not useless, it is inverted --
    which usually means the labels went in the other way round.
    """
    score_tensor, label_tensor = _as_tensors(scores, labels)
    positives = label_tensor == 1
    n_positive = int(positives.sum())
    n_negative = int(score_tensor.numel() - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise MetricError(f"AUC needs both classes present, got {n_positive} positive and {n_negative} negative")
    ranks = _average_ranks(score_tensor)
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))

def roc_curve(scores: Sequence[float], labels: Sequence[int]) -> Tuple[List[float], List[float]]:
    """False-positive and true-positive rates at every threshold the scores admit

    The curve roc_auc reports the area under, returned as two parallel lists
    running from (0, 0) to (1, 1). Ties are stepped over in one move rather
    than one at a time, so a probe that gives many examples the same score
    produces a diagonal segment there instead of a staircase that would
    overstate how finely it can separate them.
    """
    score_tensor, label_tensor = _as_tensors(scores, labels)
    positives = int((label_tensor == 1).sum())
    negatives = int(score_tensor.numel() - positives)
    if positives == 0 or negatives == 0:
        raise MetricError(
            f"a ROC curve needs both classes present, got {positives} positive and {negatives} negative"
        )

    order = torch.argsort(score_tensor, descending=True)
    ranked_scores = score_tensor[order]
    ranked_labels = label_tensor[order]

    false_positive_rates = [0.0]
    true_positive_rates = [0.0]
    true_positives = false_positives = 0
    index = 0
    while index < ranked_scores.numel():
        threshold = ranked_scores[index]
        while index < ranked_scores.numel() and ranked_scores[index] == threshold:
            if ranked_labels[index] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        true_positive_rates.append(true_positives / positives)
        false_positive_rates.append(false_positives / negatives)
    return false_positive_rates, true_positive_rates

def accuracy(scores: Sequence[float], labels: Sequence[int], threshold: float = 0.0) -> float:
    """Fraction correct when everything above the threshold is called positive"""
    score_tensor, label_tensor = _as_tensors(scores, labels)
    return float(((score_tensor > threshold).to(torch.int64) == label_tensor).to(torch.float64).mean())

def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
    """The threshold with the highest accuracy, and that accuracy

    Chosen on whatever data you pass in, so choosing it on the test set is how
    a reported number stops meaning anything. Pick it on validation.
    """
    score_tensor, label_tensor = _as_tensors(scores, labels)
    candidates = torch.unique(score_tensor)
    midpoints = torch.cat([candidates[:1] - 1.0, (candidates[:-1] + candidates[1:]) / 2, candidates[-1:] + 1.0])
    scored = [(accuracy(score_tensor, label_tensor, float(point)), float(point)) for point in midpoints]
    best_accuracy, best_point = max(scored)
    return best_point, best_accuracy

def logit_difference(
    logits: torch.Tensor, positive: Sequence[int], negative: Sequence[int]
) -> torch.Tensor:
    """logit(positive) - logit(negative) for each row, given one token pair per row

    The behaviour metric for a two-answer task. It is a difference rather than
    a probability on purpose: the softmax's normalizer is shared by both
    tokens, so it cancels here, and what is left moves only when the model
    actually shifts weight between the two answers. An accuracy over the same
    pairs throws away how close the call was.
    """
    values = torch.as_tensor(logits, dtype=torch.float64)
    if values.dim() != 2:
        raise MetricError(f"logits must be [batch, vocab], got shape {tuple(values.shape)}")
    if len(positive) != values.shape[0] or len(negative) != values.shape[0]:
        raise MetricError(
            f"{values.shape[0]} rows of logits but {len(positive)} positive and "
            f"{len(negative)} negative token ids; they index the same batch"
        )
    rows = torch.arange(values.shape[0])
    return values[rows, torch.as_tensor(list(positive))] - values[rows, torch.as_tensor(list(negative))]

def recovery(patched: float, clean: float, corrupted: float) -> float:
    """Where a patched run sits between the corrupted and the clean baseline, as 0 to 1

    1.0 means this intervention alone restored the clean behaviour; 0.0 means
    it changed nothing. Values outside [0, 1] are real results, not errors: a
    component can overshoot, and a negative one is a component whose clean
    value actively works against the answer.

    The two baselines have to come from the same prompts as the patched run,
    or the scale means nothing. When they coincide -- a prompt the corruption
    did not actually corrupt -- there is no span to normalize against, and
    saying 0 is more honest than dividing by it.
    """
    span = clean - corrupted
    if span == 0:
        return 0.0
    return (patched - corrupted) / span

def spearman(first: Sequence[float], second: Sequence[float]) -> float:
    """Rank correlation between two scorings of the same things

    Two circuit-finding techniques disagree about magnitudes by construction:
    one reports logits along the direct path, the next reports a fraction of a
    corruption's span, and a third reports a first-order estimate of the
    second. None of those are on the same scale, so the question that can
    honestly be asked of all of them is whether they put the components in the
    same order.

    Ties share a rank, which is what stops a technique that scores half the
    heads at exactly zero from being credited with an ordering it never made.
    """
    left = torch.as_tensor(first, dtype=torch.float64).flatten()
    right = torch.as_tensor(second, dtype=torch.float64).flatten()
    if left.numel() != right.numel():
        raise MetricError(f"{left.numel()} scores against {right.numel()}; a correlation ranks the same things")
    if left.numel() < 2:
        raise MetricError("a rank correlation needs at least two items to order")
    ranked_left = _average_ranks(left) - _average_ranks(left).mean()
    ranked_right = _average_ranks(right) - _average_ranks(right).mean()
    scale = ranked_left.norm() * ranked_right.norm()
    if scale == 0:
        raise MetricError("one of the scorings is constant, so it states no order to correlate against")
    return float((ranked_left * ranked_right).sum() / scale)

def jaccard(first: Iterable, second: Iterable) -> float:
    """How much two sets share, as the intersection over the union

    The overlap number for two circuits. It is reported rather than a raw
    intersection count because two techniques that each keep eight heads and
    two that each keep eighty are not the same claim about agreement.

    Read it against what chance would give: two independent selections of a
    K share of the same components overlap at about K / (2 - K), which is
    ~5% at K = 10%. An overlap that beats chance is the finding; one that
    does not is two techniques agreeing only that a model has heads.
    """
    left, right = set(first), set(second)
    if not left and not right:
        raise MetricError("both sets are empty, so their overlap is 0/0 rather than 1")
    return len(left & right) / len(left | right)

@dataclass(frozen=True)
class Cost:
    """What one run of something cost, in the units a deployment decision needs"""
    seconds: float
    items: int

    @property
    def ms_per_item(self) -> float:
        return 1000.0 * self.seconds / self.items if self.items else float("nan")

    def __str__(self) -> str:
        return f"{self.seconds:.3f}s for {self.items} items ({self.ms_per_item:.2f} ms each)"

@contextmanager
def measure(items: int) -> Iterator[List[Cost]]:
    """Time a block and report it per item

    Yields a one-element list that the Cost lands in on exit, so the caller
    can read it after the block: `with measure(n) as cost: ...` then cost[0].
    """
    started = time.perf_counter()
    holder: List[Cost] = []
    try:
        yield holder
    finally:
        holder.append(Cost(seconds=time.perf_counter() - started, items=items))

DEGENERACY_TYPES = 0.30   # unique tokens / total tokens below this is repetition, not translation
DEGENERACY_SHARE = 0.50   # or one token taking this much of the output on its own
DEGENERACY_DUPLICATE = 0.02  # or the identical output on this share of the corpus

def degeneracy(hypotheses: Sequence[str]) -> float:
    """The fraction of outputs that have stopped being language

    Deliberately crude and deliberately automatic. The checklist item it
    implements is "read ten generations at the largest ablation and stop if
    they are not language"; a human still should, but a sweep that runs
    unattended needs the machine to raise its hand. It lives here rather than
    beside any one experiment because every arm of every ablation wants it,
    and a second copy would be a second threshold to disagree with the first.
    """
    if not hypotheses:
        return 0.0
    # Collapse is not always visible inside one hypothesis. A model ablated past
    # the frontier answers 'the' to all 200 sources: each one is three tokens or
    # fewer, so the repetition rules below never examine it, and the corpus
    # scored 8.5% broken while BLEU said 0.01. The signal is only there across
    # hypotheses -- 200 distinct news sentences do not share a translation --
    # so the same text landing on a share of the corpus is counted as collapse
    # whatever its length. The floor of 3 keeps a short list from making any
    # coincidence fatal, and the healthy arms of this project peak at 2.
    repeats = Counter(normalized for text in hypotheses if (normalized := " ".join(text.split()).casefold()))
    duplicate_limit = max(3, round(DEGENERACY_DUPLICATE * len(hypotheses)))
    broken = 0
    for text in hypotheses:
        tokens = text.split()
        # an empty generation is the most degenerate outcome there is, and an
        # early version scored it 0.0 by falling through the too-short guard
        # below -- a matched random ablation that silenced the model entirely
        # was reported as perfectly healthy
        if not tokens:
            broken += 1
            continue
        # punctuation with no letters in it is not a translation at any length:
        # '...', '(', '(   (   (' all scored clean under a rule that only looked
        # for repetition among four or more tokens
        if not any(character.isalpha() for character in text):
            broken += 1
            continue
        if repeats[" ".join(tokens).casefold()] >= duplicate_limit:
            broken += 1
            continue
        if len(tokens) < 4:
            continue
        counts = Counter(tokens)
        if (len(counts) / len(tokens) < DEGENERACY_TYPES
                or counts.most_common(1)[0][1] / len(tokens) > DEGENERACY_SHARE):
            broken += 1
    return round(broken / len(hypotheses), 3)

def benjamini_hochberg(pvalues: Dict[str, float]) -> Dict[str, float]:
    """FDR-corrected q-values, because 18 tests at alpha .05 buy ~1 false positive for free

    Reported beside the raw p rather than instead of it. The raw value answers
    "would this component look real if it were the only one tested", which is
    the question a reader of a single row asks; the q-value answers "does it
    look real given that 18 rows were screened to find it", which is the
    question the sweep actually poses. They disagree here, and a table showing
    only one of them would be arguing for a conclusion rather than reporting.
    """
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    qvalues: Dict[str, float] = {}
    running = 1.0
    for rank in range(total, 0, -1):
        name, pvalue = ordered[rank - 1]
        running = min(running, pvalue * total / rank)
        qvalues[name] = round(running, 5)
    return qvalues
