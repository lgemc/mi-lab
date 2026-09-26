from typing import List, Sequence

from ....core.metrics import degeneracy
from ....telemetry.observe import Progress, log
from ..data.translation import clean_completion

"""
Generating sentences from an ablated model, and looking at them before believing a score.

`methods/knockout/ablate.py` takes a component out; this asks the model what it
says afterwards. The split is the second axis and it is a thin cut with a
reason on both sides. Mean ablation of a component is arithmetic at a site and
does not care what comes out; a *completion* is text, cleaned by a rule about
where a translation ends and the drift begins, and read by a person who can
tell language from a repeated token. One of those two is about language.

`preview` is the half that is not optional and not behind a flag. The checklist
item is "read ten generations at the largest ablation and stop if they are not
language", and the run that skipped it reported a threshold pass by a factor of
32 on a model emitting a single repeated token.

A common pipe could be: ablate | translate | preview | bleu
"""

GENERATION_CHUNK = 100    # sentences per progress tick when translating
MAX_NEW_TOKENS = 64       # a WMT sentence, with room to spare

PREVIEW_SAMPLES = 3
PREVIEW_WIDTH = 96


def translate(adapter, prompts: Sequence[str], label: str = "translate", chunk: int = GENERATION_CHUNK,
              max_new_tokens: int = MAX_NEW_TOKENS) -> List[str]:
    """Generate one completion per prompt, chunked here rather than in the adapter so the pass reports progress

    `adapter.generate` batches internally and returns only when every prompt
    is done, which makes a 200-sentence pass a single silent minute. Chunking
    at this level produces the same completions in the same order and lets
    the loop say where it is.
    """
    chunks = [prompts[start : start + chunk] for start in range(0, len(prompts), chunk)]
    bar = Progress(len(chunks), label, indent=2)
    done: List[str] = []
    for piece in chunks:
        done.extend(clean_completion(text) for text in adapter.generate(list(piece), max_new_tokens=max_new_tokens))
        bar.tick(f"{len(done)}/{len(prompts)} sentences")
    return done

def preview(hypotheses: Sequence[str], label: str = "sample", count: int = PREVIEW_SAMPLES,
            indent: int = 1) -> float:
    """Log a few actual generations, because a metric cannot show you a broken model

    Not optional and not behind a flag. The checklist item is "read ten
    generations at the largest ablation and stop if they are not language",
    and the run that skipped it reported a threshold pass by a factor of 32 on
    a model emitting a single repeated token. Returns the degeneracy share so
    the caller can record it beside the score.
    """
    if not hypotheses:
        log(f"{label}: no generations at all -- the pass produced nothing", indent=indent)
        return 1.0
    for index, text in enumerate(hypotheses[:count]):
        shown = text if len(text) <= PREVIEW_WIDTH else text[:PREVIEW_WIDTH - 1] + "…"
        log(f"{label}[{index}]: {shown!r}", indent=indent)
    empty = sum(1 for text in hypotheses if not text.strip())
    if empty:
        log(f"{label}: {empty}/{len(hypotheses)} generations are empty", indent=indent)
    broken = degeneracy(hypotheses)
    if broken:
        log(f"{label}: DEGENERACY {broken:.1%} of generations are repeated tokens, not language -- "
            "this is a broken model, not an ablated one", indent=indent)
    return broken
