"""Neutral prose for measurements that need text no checkpoint was tuned on"""

from typing import List


def passages(count: int, min_chars: int = 1200) -> List[str]:
    """Long stretches of ordinary prose, for measuring a variance over

    Wikitext rather than this study's translation corpus, and long rather than
    short, for two different reasons. Long because FVU divides by the spread of
    a layer's MLP output across positions, and over a six-token prompt there is
    barely any spread to divide by -- the same dictionary reports an FVU half
    again as large on short prompts as on these. Neutral prose because the two
    checkpoints being compared are a base model and an instruction-tuned one,
    and any text shaped like an instruction is text one of them was trained on
    and the other was not.

    Moved here from `scripts/phase2_diffing.py` when `phase2_kl` became its
    second caller.
    """
    from datasets import load_dataset

    rows = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"]
    return [row for row in rows if len(row) > min_chars][:count]
