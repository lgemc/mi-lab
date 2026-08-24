import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import torch

"""
Every experiment in this framework is supposed to end in a number, and these
are the numbers. AUC and accuracy say whether a method works; Cost says what
it costs to run, which is the half that decides whether it ever ships.

AUC is computed rank-wise with ties averaged rather than by thresholding a
grid, so it is exact and does not quietly depend on how the scores happen to
be scaled.

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
