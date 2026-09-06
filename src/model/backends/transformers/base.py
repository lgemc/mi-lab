"""
The adapter's state, and the plumbing every measurement above it is built from.

What lives here is what more than one question needs: the resolved config, the
three module lists a hook site is looked up in, and the five helpers that turn
a list of prompts into forward passes -- encode to one length, walk it in
batch_size chunks, record the residual stream, record the heads, split them
apart. A method belongs in this file when two of the mixins call it and in its
own file otherwise.

The two recording contexts are the pair of sites the whole backend addresses:
a decoder block's output, and the input to the projection that mixes the heads
back in. Recording hooks are registered after any intervention hook, and
PyTorch runs forward hooks in registration order while threading the modified
output through them, so capturing at a steered layer sees the intervention
rather than the value it replaced.

A common pipe could be: _encode | _chunks | _record
"""

from contextlib import contextmanager
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Sequence

import torch

from ....core.config import ConfigError, ModelConfig
from .layout import _attention_projection, _blocks, _mlp
from .patching import _Patch


class AdapterBase:
    """The model, the tokenizer, the sites, and the batching that reaches them"""

    def __init__(self, cfg: ModelConfig, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.blocks = _blocks(model)
        self.projections = [_attention_projection(block, index) for index, block in enumerate(self.blocks)]
        self.mlps = [_mlp(block, index) for index, block in enumerate(self.blocks)]
        self.cfg = replace(
            cfg.with_sizes(
                n_layers=len(self.blocks),
                d_model=model.config.hidden_size,
                n_heads=getattr(model.config, "num_attention_heads", None),
            ),
            # stamped in the way the sizes are, and for the same reason: "auto" is a
            # question, and everything downstream that prints or records a device wants
            # the answer. The weights are already on it by the time this runs
            device=str(model.device).split(":")[0],
        )
        self._patch: Optional[_Patch] = None

    def layer(self, frac: Optional[float] = None) -> int:
        """Resolve a depth fraction to an absolute layer index"""
        return self.cfg.layer(frac)

    def _resolve_layers(self, layers: Optional[Sequence[int]]) -> List[int]:
        """Default to the config's probe layer, and reject indices this model does not have"""
        if layers is None:
            return [self.layer()]
        layers = list(layers)
        if not layers:
            raise ConfigError("capture needs at least one layer")
        out_of_range = [index for index in layers if not 0 <= index < self.cfg.n_layers]
        if out_of_range:
            raise ConfigError(f"layers {out_of_range} are outside the {self.cfg.n_layers} layers of '{self.cfg.id}'")
        return layers

    def _chunks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Walk the prompts in batch_size chunks, telling any active patch which rows it is on"""
        for start in range(0, len(input_ids), self.cfg.batch_size):
            ids = input_ids[start : start + self.cfg.batch_size].to(self.model.device)
            mask = attention_mask[start : start + self.cfg.batch_size].to(self.model.device)
            if self._patch is not None:
                self._patch.select(start, ids.shape[0])
            yield ids, mask

    @contextmanager
    def _record(self, layers: Sequence[int]) -> Iterator[Dict[int, torch.Tensor]]:
        """Collect the residual stream leaving each requested block

        Recording hooks are registered after any steering hook, and PyTorch
        runs forward hooks in registration order while threading the modified
        output through them, so capturing at a steered layer sees the
        intervention rather than the value it replaced.
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(index: int):
            def hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden.detach()
            return hook

        for index in layers:
            handles.append(self.blocks[index].register_forward_hook(make_hook(index)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _encode(self, prompts: Sequence[str], padding_side: str):
        """Tokenize every prompt to one uniform length, so chunks stay comparable"""
        self.tokenizer.padding_side = padding_side
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        return encoded["input_ids"], encoded["attention_mask"]

    @contextmanager
    def _record_heads(self, layers: Sequence[int], detach: bool = True) -> Iterator[Dict[int, torch.Tensor]]:
        """Collect what each head wrote, before the projection mixed them together

        detach=False keeps the recorded tensors attached to the graph that
        produced them, which is what a gradient at this site needs. It is off
        by default because everything else here reads activations under
        no_grad, and a capture that quietly held a backward graph would keep
        the whole forward pass alive in memory.
        """
        captured: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(index: int):
            def hook(module, args):
                captured[index] = args[0].detach() if detach else args[0]
            return hook

        for index in layers:
            handles.append(self.projections[index].register_forward_pre_hook(make_hook(index)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def _split_heads(self, merged: torch.Tensor) -> torch.Tensor:
        """Split a [..., n_heads * d_head] tensor into [..., n_heads, d_head]"""
        return merged.reshape(*merged.shape[:-1], self.cfg.n_heads, merged.shape[-1] // self.cfg.n_heads)
