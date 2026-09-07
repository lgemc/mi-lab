"""
Ablating one edge toward a counterfactual run, leaving the source's others alone.

A node ablation deletes a component's output everywhere at once. An edge
ablation deletes what one *reader* sees of it, which is a strictly finer
intervention and a different claim: a head that matters through one path and
not another is indistinguishable from an unimportant head under node ablation,
and separable under this.

Because the residual stream is a sum, an edge is a term in it and ablating one
is arithmetic on the destination's input:

    input := input + (off_write_source - live_write_source)

The live write is subtracted rather than the captured clean one, so edits
compose -- a destination late in the stack reads sources that earlier edits
already changed, and freezing them at their un-edited values would silently
make every edge independent of every other.

A common pipe could be: residual_sources | edge_patch | logits
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Sequence, Tuple

import torch

from ....core.config import ConfigError
from .layout import _attention_norm, _mlp_norm


class EdgePatchMixin:
    """One edge's counterfactual, applied at the destination that reads it"""

    @contextmanager
    def edge_patch(self, off: Dict[str, torch.Tensor], edges: Sequence[Tuple[str, str]]) -> Iterator[None]:
        """Ablate individual (source, destination) edges, leaving the source's other edges alone

        A node ablation deletes a component's output everywhere at once. An
        edge ablation deletes what one *reader* sees of it, which is a strictly
        finer intervention and a different claim: a head that matters through
        one path and not another is indistinguishable from an unimportant head
        under node ablation, and separable under this.

        Because the residual stream is a sum, an edge is a term in it, and
        ablating one is arithmetic on the destination's input:

            input := input + (off_write_source - live_write_source)

        The live write is subtracted rather than the captured clean one, so
        edits compose: a destination late in the stack reads sources that
        earlier edits already changed, and freezing them at their un-edited
        values would silently make every edge independent of every other.

        Source ids are `embed`, `head:L:H`, `mlp:L`, `bias:L`; destinations are
        `attn:L` and `mlp:L`. An edge whose source is not upstream of its
        destination is refused rather than ignored -- attention at layer L
        cannot read layer L's own MLP, and an experiment that thought it could
        is not one whose other edges should be trusted.
        """
        wanted: Dict[str, List[str]] = {}
        for source, destination in edges:
            self._check_edge(source, destination)
            wanted.setdefault(destination, []).append(source)
        if not wanted:
            yield
            return

        merged: Dict[int, torch.Tensor] = {}
        mlp_out: Dict[int, torch.Tensor] = {}
        # the live embedding, which is what block 0 is handed. Without it `embed`
        # had no live value to subtract and its counterfactual was added on top of
        # the real one -- the strict receipt (all edges into a destination must
        # equal replacing that destination's input) is what caught it, at a delta
        # of 2.16 where float noise was expected.
        embedded: Dict[int, torch.Tensor] = {}
        handles = []
        # Reconstructing a head's write calls the projection again, which re-enters
        # the pre-hook capturing that projection's input -- so without this the
        # first call (the bias, on a zero vector) overwrites the merged heads and
        # every head after it is computed from zeros. The identity check caught it:
        # ablating an edge toward the run's own values moved the logits by 2.5.
        reconstructing = {"busy": False}

        def capture_embedding(module, args):
            if not reconstructing["busy"]:
                embedded[0] = args[0].detach()

        def capture_merged(index):
            def hook(module, args):
                if not reconstructing["busy"]:
                    merged[index] = args[0].detach()
            return hook

        def capture_mlp(index):
            def hook(module, args, output):
                if not reconstructing["busy"]:
                    mlp_out[index] = (output[0] if isinstance(output, tuple) else output).detach()
            return hook

        def destination_hook(destination):
            sources = wanted[destination]
            def hook(module, args):
                residual = args[0]
                if off["embedding"].shape[1] != residual.shape[1]:
                    raise ConfigError(
                        f"the counterfactual run is {off['embedding'].shape[1]} positions and this one is "
                        f"{residual.shape[1]}. An edge is a term in one position's residual sum, so the two "
                        "runs have to be length-aligned; pad or pick prompts that tokenize alike."
                    )
                delta = torch.zeros_like(residual)
                reconstructing["busy"] = True
                try:
                    for source in sources:
                        live = self._live_write(source, merged, mlp_out, embedded, residual)
                        delta = delta + (
                            self._off_write(source, off).to(residual.device, residual.dtype) - live)
                finally:
                    reconstructing["busy"] = False
                return (residual + delta, *args[1:])
            return hook

        # registered first, so it records what block 0 received before any
        # destination hook on that same block has edited it
        handles.append(self.blocks[0].register_forward_pre_hook(capture_embedding))
        for index in range(self.cfg.n_layers):
            handles.append(self.projections[index].register_forward_pre_hook(capture_merged(index)))
            handles.append(self.mlps[index].register_forward_hook(capture_mlp(index)))
        for destination in wanted:
            kind, layer = destination.split(":")
            norm = (_attention_norm if kind == "attn" else _mlp_norm)(self.blocks[int(layer)], int(layer))
            handles.append(norm.register_forward_pre_hook(destination_hook(destination)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
