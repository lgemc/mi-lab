import math
import re
from collections import Counter
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns

from .style import PALETTE, class_colors, set_style

"""
Visualize a LabeledPrompts before anything is trained on it, because most of
the ways a probe result turns out to be worthless are visible here and nowhere
later.

Three of the four charts exist to catch a specific failure:

- balance, because an AUC quoted without it can be a probe that learned the
  base rate;
- length by class, because if the classes differ in length the probe can read
  the length instead of the property, and last-token capture makes that easy;
- token log-odds, because a dataset separable by single words is separable by
  bag-of-words, and a linear probe clearing that bar has proved nothing about
  the model. mi-lab's own synthetic set fails this one on purpose, and the
  chart is how you see by how much.
- leakage, because a template set shuffled at random puts near-identical
  sentences on both sides of the split.

A common pipe could be: synthetic | plot_token_log_odds | plot_split_leakage
"""

def _names(dataset) -> Tuple[str, str]:
    """The dataset's own names for class 0 and 1, or the generic pair

    label_names is optional on LabeledPrompts, so this reads it defensively
    rather than requiring every dataset to carry one.
    """
    names = getattr(dataset, "label_names", None)
    return tuple(names) if names else ("negative", "positive")

def words(text: str) -> List[str]:
    """Lowercase word tokens, the crude tokenization a bag-of-words baseline would use

    Deliberately not the model's tokenizer: the question this answers is
    whether a word-level baseline could do the job, so it should use a
    word-level view.
    """
    return re.findall(r"[a-z0-9']+", text.lower())

def plot_class_balance(dataset, ax=None):
    """Bar chart of how many examples each class has"""
    set_style()
    negative_name, positive_name = _names(dataset)
    counts = [len(dataset) - dataset.positives, dataset.positives]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.barplot(x=[negative_name, positive_name], y=counts, ax=ax,
                hue=[negative_name, positive_name], legend=False,
                palette=class_colors((negative_name, positive_name)))
    for index, count in enumerate(counts):
        ax.text(index, count, f"{count}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Examples")
    ax.set_title(f"{dataset.name}: n={len(dataset)}, {dataset.balance:.0%} {positive_name}")
    return ax

def plot_length_distribution(dataset, tokenize: Optional[Callable[[str], Sequence]] = None, ax=None):
    """Overlaid histograms of example length per class

    Pass the model's tokenizer through `tokenize` to count real tokens; the
    default counts words, which is enough to see a difference between the
    classes but is not what the model saw.
    """
    set_style()
    tokenize = tokenize or words
    negative_name, positive_name = _names(dataset)
    colors = class_colors((negative_name, positive_name))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    means = {}
    for label, name in ((0, negative_name), (1, positive_name)):
        lengths = [len(tokenize(text)) for text, value in zip(dataset.texts, dataset.labels, strict=True) if value == label]
        means[name] = sum(lengths) / len(lengths) if lengths else float("nan")
        sns.histplot(lengths, bins=20, ax=ax, color=colors[name], label=name, alpha=0.55, element="step")

    ax.legend(title="Class")
    ax.set_xlabel("Tokens per example")
    ax.set_ylabel("Examples")
    gap = abs(means[positive_name] - means[negative_name])
    ax.set_title(f"Length by class (means {means[negative_name]:.1f} vs {means[positive_name]:.1f}, gap {gap:.1f})")
    return ax

def token_log_odds(dataset, tokenize: Optional[Callable[[str], Sequence]] = None, alpha: float = 1.0):
    """Smoothed log-odds of every token between the two classes, most positive first

    Add-alpha smoothing over the union vocabulary, so a token appearing in one
    class only gets a large finite score rather than an infinite one.
    """
    tokenize = tokenize or words
    counters = (Counter(), Counter())
    for text, label in zip(dataset.texts, dataset.labels, strict=True):
        counters[label].update(tokenize(text))

    vocabulary = set(counters[0]) | set(counters[1])
    totals = [sum(counter.values()) for counter in counters]
    scores = []
    for token in vocabulary:
        rates = [
            (counters[label][token] + alpha) / (totals[label] + alpha * len(vocabulary))
            for label in (0, 1)
        ]
        scores.append((token, math.log(rates[1] / rates[0])))
    return sorted(scores, key=lambda pair: pair[1], reverse=True)

def plot_token_log_odds(dataset, k: int = 12, tokenize: Optional[Callable[[str], Sequence]] = None, ax=None):
    """Diverging bar chart of the k tokens most predictive of each class

    This is the bag-of-words baseline made visible. If a handful of words
    separate the classes cleanly, a linear probe reaching AUC 1.0 has not
    shown that the model represents the property -- it has shown the property
    was written on the surface of the text.
    """
    set_style()
    negative_name, positive_name = _names(dataset)
    scores = token_log_odds(dataset, tokenize=tokenize)
    selected = scores[:k] + scores[-k:]
    labels = [token for token, _ in selected]
    values = [value for _, value in selected]
    colors = [PALETTE["positive"] if value > 0 else PALETTE["negative"] for value in values]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.32 * len(labels) + 1.5))

    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["grid"], linewidth=1)
    ax.set_xlabel(f"log odds  <- {negative_name}   |   {positive_name} ->")
    ax.set_ylabel("Token")
    ax.set_title(f"Most class-predictive tokens in {dataset.name}")
    return ax

def _jaccard(left: set, right: set) -> float:
    """Overlap of two token sets, 1.0 when identical"""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)

def plot_split_leakage(train, test, tokenize: Optional[Callable[[str], Sequence]] = None, ax=None):
    """Histogram of how similar each test example is to its nearest training example

    Exact duplicates across the split are counted in the title. Mass piled up
    near 1.0 without exact duplicates is the subtler version of the same
    problem: a template set where a test sentence differs from a training one
    by a single word, so the reported number is measuring memorization of a
    template rather than generalization.
    """
    set_style()
    tokenize = tokenize or words
    train_sets = [set(tokenize(text)) for text in train.texts]
    train_texts = set(train.texts)

    nearest = []
    duplicates = 0
    for text in test.texts:
        if text in train_texts:
            duplicates += 1
        tokens = set(tokenize(text))
        nearest.append(max((_jaccard(tokens, other) for other in train_sets), default=0.0))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    sns.histplot(nearest, bins=20, binrange=(0, 1), ax=ax, color=PALETTE["accent"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Jaccard similarity to nearest training example")
    ax.set_ylabel("Test examples")
    ax.set_title(f"Split leakage: {duplicates} exact duplicate(s) of {len(test)} test examples")
    return ax
