from typing import Dict, Tuple

"""
What every number this lab reports actually is, in one table.

A metric name is not a definition -- `faithfulness` names a logit-difference
recovery here and a normalized KL reproduction elsewhere -- so v0.2 refuses a
metric that ships without one. This is where the ones this repository writes
live, in a module with no imports so that both the converters that produce
artifacts and the migration that repairs old ones can read the same table
rather than each keeping a copy that drifts.

Anything not here is a metric some other tool named, and the honest answer for
it is that its definition was never recorded -- not a plausible-sounding
sentence invented after the fact.

A common pipe could be: define | Metric | validate
"""

# name -> (what was computed, what the number is in)
DEFINITIONS: Dict[str, Tuple[str, str]] = {
    "faithfulness": (
        "recovery of the clean logit difference when only these heads are restored into the "
        "corrupted run; 1.0 is the clean baseline, 0.0 the corrupted one",
        "recovery",
    ),
    "necessity": (
        "1 - the recovery left when these heads alone are corrupted in an otherwise clean run; "
        "1.0 means the rest of the model does not do this without them",
        "recovery",
    ),
    "n_heads": ("how many heads the greedy search kept", "heads"),
    "threshold": (
        "the smallest gain in cumulative recovery for which the search adds another head",
        "recovery",
    ),
    "attribution_remainder": (
        "logit difference left over after summing every component's direct write through the frozen "
        "unembedding; a receipt on the decomposition, not a result",
        "logits",
    ),
    "auc": ("area under the ROC curve of the probe's score against the held-out labels", "auc"),
    "accuracy": ("share of held-out examples the probe scores on the correct side of zero", "share"),
    "n": ("held-out examples the other metrics were measured over", "examples"),
    "train_loss": ("final training objective on the fitting set, not a held-out number", "loss"),
    "norm": ("L2 norm of the direction as stored, before it is scaled by a strength", "activation"),
    "max_strength": (
        "the largest strength the sweep visited, which is a range that was searched and not a "
        "ceiling that was found",
        "mean activation norms",
    ),
}

UNKNOWN = "unspecified"

def describe(name: str, source: str = "") -> Tuple[str, str]:
    """The definition and unit for a metric name, or an honest admission

    A name this table does not know gets a definition saying so rather than a
    guess. That keeps the v0.2 gate meaningful -- the field says whether the
    quantity is pinned down, and "nobody wrote it down" is a real answer to
    that question while an invented sentence is not.
    """
    if name in DEFINITIONS:
        return DEFINITIONS[name]
    trailer = f" by '{source}'" if source else ""
    return (f"not recorded{trailer}; this metric shipped as a bare number and its definition is unknown", UNKNOWN)
