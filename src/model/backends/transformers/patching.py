"""
Writing another run's activations in, at exactly the sites capture reads.

A patch is the causal half of every circuit measurement here, and it is a
measurement only because the two sites match: the residual stream leaving a
block and the input to the attention output projection are read by `capture`
and `head_outputs` and written by nothing else, so patching a site with what
was already there is exactly a no-op. Every causal number in this framework is
a difference against that no-op.

`_Patch` is the other half of the contract: donor tensors cover the whole
prompt list while a patched forward pass still runs in batch_size chunks, so
every hook has to slice the rows of the chunk it fires inside.

A common pipe could be: head_outputs | patch | logits
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional

import torch

from ....core.config import ConfigError


@dataclass
class _Patch:
    """Donor activations to write in, and which rows of them the current chunk needs

    A patched forward pass still runs in batch_size chunks, and the donor
    tensors cover the whole prompt list, so every hook has to slice the rows
    belonging to the chunk it is firing inside. Handing a hook the full donor
    would patch the wrong prompts as soon as one batch becomes two.
    """
    residual: Dict[int, torch.Tensor] = field(default_factory=dict)
    heads: Dict[int, Dict[int, torch.Tensor]] = field(default_factory=dict)
    start: int = 0
    stop: int = 0

    def select(self, start: int, rows: int) -> None:
        self.start, self.stop = start, start + rows

    def rows(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.start : self.stop]

class PatchMixin:
    """Overwriting a run's activations with another run's, under hooks removed on exit"""

    @contextmanager
    def patch(
        self,
        residual: Optional[Dict[int, torch.Tensor]] = None,
        heads: Optional[Dict[int, Dict[int, torch.Tensor]]] = None,
    ) -> Iterator[None]:
        """Overwrite activations with values from another run, for the duration of the block

        residual maps a layer to a [batch, seq, d_model] replacement for the
        stream *leaving* that block -- the same site capture() reads, so a
        capture from one run drops straight into a patch of another. heads maps
        a layer to {head: [batch, seq, d_head]}, replacing only those heads and
        leaving the layer's others alone.

        Donor tensors cover the whole prompt list; the hooks slice out the rows
        of whichever chunk is running. Patches do not nest: a second one while
        the first is open would silently decide an ordering nobody chose.
        """
        if self._patch is not None:
            raise ConfigError("a patch is already active; build one patch describing every site instead of nesting")

        active = _Patch(residual=dict(residual or {}), heads=dict(heads or {}))
        for index in list(active.residual) + list(active.heads):
            self._resolve_layers([index])
        handles = []

        def make_residual_hook(index: int):
            def hook(module, args, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output
                donor = active.rows(active.residual[index]).to(hidden.device, hidden.dtype)
                return (donor, *output[1:]) if is_tuple else donor
            return hook

        def make_head_hook(index: int):
            def hook(module, args):
                merged = self._split_heads(args[0]).clone()
                for head, value in active.heads[index].items():
                    merged[:, :, head] = active.rows(value).to(merged.device, merged.dtype)
                return (merged.reshape(*args[0].shape), *args[1:])
            return hook

        for index in active.residual:
            handles.append(self.blocks[index].register_forward_hook(make_residual_hook(index)))
        for index in active.heads:
            handles.append(self.projections[index].register_forward_pre_hook(make_head_hook(index)))

        self._patch = active
        try:
            yield
        finally:
            self._patch = None
            for handle in handles:
                handle.remove()
