"""Loading a checkpoint with its MLPs replaced by a transcoder, for whoever asks.

This was `scripts/phase1c_graphs.py`'s private helper until a second script
needed it. Two scripts now do -- the attribution graphs, and the dictionary-fit
half of `phase2_diffing` -- so it lives here under a test rather than in
whichever script happened to want it first.

Moving it drew out an interface the private version did not have: it takes a
checkpoint name and a dictionary release **separately** rather than one config.
That is not tidying. A config's `transcoder:` block records the dictionary
fitted *for that checkpoint*, and the question "does a dictionary still describe
a checkpoint it was not fitted on" is asked by pairing the two deliberately. A
function that could only read both out of one config could not ask it, and a
config invented to hold a pairing would be recording a fit that nobody
measured.

Telemetry stays out: `src/model/` sits below it and imports nothing from
`telemetry`, so progress is reported through the `note` and `attempt` callbacks
and the caller decides what a log line looks like.

A common pipe could be: register | load_replacement | setup_attribution
"""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Dict, Optional, Tuple

from ..core.config import ConfigError

BACKENDS = ("transformerlens", "nnsight")


class ReplacementError(ConfigError):
    """No backend could put this checkpoint and this dictionary together"""


def register(hf_name: str) -> bool:
    """Teach TransformerLens a checkpoint it does not list but can already load

    This is what a transcoder's model card means by "requires a patched
    Transformer Lens", and the patch is smaller than it sounds: TransformerLens
    3.2.1 lists `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-0.6B-Base` among its official
    models and simply does not list `Qwen/Qwen3-1.7B-Base`. The refusal is a
    name lookup, not an architecture gap.

    It is safe *there* and the reason is measured rather than assumed: those two
    checkpoints' HuggingFace configs agree on 24 of 26 fields, differing only in
    `eos_token_id` (a base model has no chat turn to end) and
    `max_position_embeddings`. Neither changes a weight shape, and
    TransformerLens reads the rest off the checkpoint.

    Returns whether it had to do anything, so the caller can say so. A future
    TransformerLens that ships the name makes this a no-op rather than a
    conflict.
    """
    import transformer_lens.loading_from_pretrained as loading

    if hf_name in loading.OFFICIAL_MODEL_NAMES:
        return False
    loading.OFFICIAL_MODEL_NAMES.append(hf_name)
    return True


def load_replacement(
    hf_name: str,
    release: str,
    dtype: str = "bfloat16",
    backend: str = "transformerlens",
    note: Optional[Callable[[str], None]] = None,
    wrap: Optional[Callable[[str], ContextManager[Dict[str, Any]]]] = None,
) -> Tuple[object, str]:
    """One checkpoint's weights with one dictionary's features, and which backend ran

    `backend` is the thing most likely to need changing, and for the checkpoint
    this repo has a dictionary for it is already known to need it (see
    `register`). So `auto` tries TransformerLens and falls back to `nnsight`,
    which covers the architectures TransformerLens does not at the cost of speed
    and memory.

    Which one ran is returned rather than assumed, because they are two
    implementations of one forward pass and this repo has been bitten by that
    shape before (`tests/edges.py::test_both_paths_through_the_gate_agree`).

    `hf_name` and `release` are independent on purpose -- see the module
    docstring. Nothing here checks that the dictionary was fitted on this
    checkpoint, because measuring what happens when it was not is the point.

    `wrap` is a context manager per attempt, yielding a dict this function drops
    the backend and release into -- `telemetry.observe.step` is one, and the
    default is a no-op. It is a parameter rather than an import because a load
    that takes minutes should be announced before it blocks, and this layer is
    not allowed to know how announcing works.
    """
    import torch
    from circuit_tracer import ReplacementModel

    resolved = getattr(torch, dtype)
    if register(hf_name) and note is not None:
        note(
            f"registered {hf_name} with TransformerLens, which does not ship the name "
            "(its config differs from the listed sibling only in eos_token_id and context length)"
        )

    for attempt in [backend] if backend != "auto" else list(BACKENDS):
        try:
            announce = wrap(f"loading replacement model ({attempt})") if wrap else nullcontext({})
            with announce as facts:
                model = ReplacementModel.from_pretrained(hf_name, release, backend=attempt, dtype=resolved)
                facts["backend"] = attempt
                facts["transcoder"] = release
            return model, attempt
        except Exception as error:
            if backend != "auto":
                raise
            if note is not None:
                note(
                    f"{attempt} backend could not load this checkpoint ({type(error).__name__}: {error}); "
                    "trying the next"
                )
    raise ReplacementError(
        f"no circuit-tracer backend could load {hf_name} with {release}; "
        "transformerlens supports a fixed list of architectures and nnsight covers the rest"
    )
