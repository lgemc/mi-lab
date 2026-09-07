"""
The HuggingFace backend: the correctness oracle every faster backend is
checked against. It works on any causal LM on the hub and it is slow, and
those two facts are the same fact.

This module is the assembly and nothing else -- the adapter is composed here
from one mixin per question it answers, and the registration below is what
`configs/*.yaml` names when it asks for `backend: transformers`. Where a
method lives is where its question is answered:

    layout      where the sites are in this architecture -- the whole of the
                architecture knowledge, quarantined in four lookup lists
    positions   which position is the real one, once padding is in the batch
    base        the state and the plumbing more than one mixin needs
    capture     the residual stream, read at a block's output and steered there
    outputs     tokens, next-token logits, continuations
    heads       patterns, head outputs, and gradients at the projection's input
    patching    another run's activations written at the sites capture reads
    decompose   the final token split into the writes that built it
    graph       the residual stream as a graph, and which edges exist in it
    edge_patch  one edge ablated toward a counterfactual run
    edge_gate   one edge scaled by a differentiable gate, for pruning

Two sites carry every operation. The residual stream is read and written at a
decoder block's output; heads are read and written at the input to the
projection that mixes them back in. Capture and patch address the *same* two
sites, which is what makes writing back what was already there exactly a
no-op -- and every causal number in this framework is a difference against
that no-op.

This package is named after the library it wraps, which is also the registry
key `configs/*.yaml` names. `import transformers` inside it still resolves to
the installed library: Python 3 imports are absolute unless written with a
leading dot.

A common pipe could be: load_adapter | capture | head_outputs | patch
"""

from ....core.config import ModelConfig
from ....model.adapter import DTYPES, ModelAdapter, register_backend, resolve_device
from .base import AdapterBase
from .capture import CaptureMixin
from .decompose import DecomposeMixin
from .edge_gate import EdgeGateMixin
from .edge_patch import EdgePatchMixin
from .graph import GraphMixin
from .heads import HeadMixin
from .outputs import OutputMixin
from .patching import PatchMixin


class TransformersAdapter(
    CaptureMixin,
    OutputMixin,
    HeadMixin,
    PatchMixin,
    DecomposeMixin,
    GraphMixin,
    EdgePatchMixin,
    EdgeGateMixin,
    AdapterBase,
):
    """A CircuitAdapter over a HuggingFace causal LM

    Every residual stream site is a decoder block's output, read and written
    with forward hooks; every head-level site is the input to the projection
    that mixes the heads. Capture and patch address the same two sites, so
    what a patch writes in is exactly what a capture would have read out.

    The mixins share no state beyond what `AdapterBase` holds and define no
    method twice, so the order they are listed in decides nothing -- it is the
    order of the table above, which is the order a reader meets them.
    """

@register_backend("transformers")
def _build_transformers(cfg: ModelConfig) -> ModelAdapter:
    """Load a HuggingFace causal LM in eval mode, with a pad token guaranteed"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    if tokenizer.pad_token is None:
        # GPT-2 and friends ship no pad token, and batching needs one
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_name, dtype=DTYPES[cfg.dtype])
    model.to(resolve_device(cfg.device))
    model.eval()
    return TransformersAdapter(cfg, model, tokenizer)
