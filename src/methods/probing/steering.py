import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

"""
A steering study is a sweep over one number, and this is the sweep.

Adding a direction to the residual stream always does something; the question
is whether what it does is the behaviour you wanted or the model coming apart.
Those are two different measurements and a single strength cannot separate
them, which is why every steering result worth anything is a curve rather than
one generation at strength 2.0.

Two axes are measured at each strength:

- effect, the probe's score on the steered continuation. The direction that
  writes the property in is the direction that reads it back out, so the probe
  that found it is the right instrument to check it landed.
- fluency, the share of generated words that are not repeats. Steering past
  the ceiling produces mush, and mush repeats. It is a coarse proxy and it is
  honest about being one: it catches degeneration, not subtler damage.

The ceiling is where those two curves cross -- effect still climbing while
fluency falls off. Where it sits is a fact about this model; that it exists at
all is expected to survive a model swap.

random_control is the check that makes the rest mean anything. A random vector
of the same norm, injected at the same layer, moves the model too. If the real
direction does not beat it, nothing has been demonstrated.

A common pipe could be: difference_of_means | strength_sweep | plot_strength_sweep
"""

@dataclass
class SteeringPoint:
    """What one strength did: the text it produced, and the two numbers about it"""
    strength: float
    completions: List[str] = field(default_factory=list)
    effect: float = float("nan")
    fluency: float = float("nan")

def fluency(text: str) -> float:
    """Share of words in a continuation that are not repeats of an earlier one

    1.0 is text that never repeats a word, which short continuations often
    reach; a value collapsing toward 0 is the model looping, which is what
    over-steering looks like before it looks like anything else. Empty text
    scores 0, because generating nothing is a failure rather than perfect
    non-repetition.
    """
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def random_control(vector: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """A random direction with the same norm as the given one

    The control for every steering claim. Matching the norm matters: an
    intervention's size is most of its effect, so a control that is merely
    random without being the same length is not controlling for the thing
    that actually moved the model.
    """
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
    return noise / noise.norm().clamp_min(1e-12) * vector.float().norm()

def strength_sweep(
    adapter,
    prompts: Sequence[str],
    layer: int,
    vector: torch.Tensor,
    strengths: Sequence[float],
    probe=None,
    max_new_tokens: Optional[int] = None,
) -> List[SteeringPoint]:
    """Generate at every strength and measure effect and fluency at each

    Strength 0 is worth including and is included by whoever passes the
    strengths: adapter.steer registers no hook at all at zero, so that row is
    the unsteered baseline measured by exactly the same path as the rest,
    rather than a separate code path that might differ.

    The probe scores the prompt plus its continuation, not the continuation
    alone. A continuation read without its prompt is a fragment, and the probe
    was fit on whole sentences.
    """
    points = []
    for strength in strengths:
        with adapter.steer(layer, vector, strength):
            completions = adapter.generate(prompts, max_new_tokens=max_new_tokens)

        effect = float("nan")
        if probe is not None:
            full = [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions, strict=True)]
            activations = adapter.capture(full, layers=[probe.layer])
            effect = float(probe.score(activations).mean())

        points.append(SteeringPoint(
            strength=float(strength),
            completions=completions,
            effect=effect,
            fluency=sum(fluency(text) for text in completions) / len(completions),
        ))
    return points
