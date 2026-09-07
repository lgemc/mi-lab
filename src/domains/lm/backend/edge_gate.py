"""
Scaling an edge by a differentiable gate, and the second implementation that serves it.

`edge_patch` asks what happens if this edge carried different information.
Pruning asks the other question -- what if it were not there -- so the
counterfactual is zero, and with that substitution the same arithmetic
collapses into a gate:

    residual + (0 - live) * (1 - g)  ==  residual - live + g * live

g = 1 leaves the edge untouched, g = 0 removes the source's whole contribution
to that one destination, and anything between scales it.

**One equation, two implementations, chosen by torch.is_grad_enabled().**
Training accumulates one term per edge because every gate needs a node in the
graph, open ones included. Inference classifies the gates once at context entry
in a single device-to-host transfer, drops the open edges -- (1 - 1) * live is
a provable no-op and was most of the work at serving density -- and reduces the
rest with one stack().sum(), so a destination costs about three kernel launches
instead of three per edge. Two implementations of one equation is two chances
to be wrong, which is why tests/model/edges.py checks them against each other.

A common pipe could be: edges | edge_gate | prune
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import torch

from .layout import _attention_norm, _mlp_norm


class EdgeGateMixin:
    """A gate per edge, differentiable while training and constant while serving"""

    @contextmanager
    def edge_gate(self, gates: Dict[Tuple[str, str], torch.Tensor]) -> Iterator[None]:
        """Scale each (source, destination) edge by a differentiable gate, for pruning

        `edge_patch` ablates an edge *toward a counterfactual run*: the
        destination sees what the source wrote in the `off` run instead of this
        one. Pruning is the other question -- not "what if this edge carried
        different information" but "what if it were not there" -- so the
        counterfactual here is zero, and with that substitution the same
        arithmetic collapses into a gate:

            residual + (0 - live) * (1 - g)  ==  residual - live + g * live

        g = 1 leaves the edge untouched, g = 0 removes the source's whole
        contribution to that one destination, and anything between scales it.

        The gradient is the point. Every captured write is detached, so the only
        path from the loss back to `g` is the multiplication above -- but the
        edited residual feeds every later block, so the gate is trained on its
        real downstream effect and not on a first-order stand-in.

        Reuses `edge_patch`'s helpers rather than its body, including the
        `reconstructing` latch: rebuilding a head's write calls the projection
        again, which re-enters the pre-hook that captures that projection's
        input, and without the latch the first reconstruction overwrites the
        merged heads and every head after it is computed from zeros.
        """
        wanted: Dict[str, List[str]] = {}
        for (source, destination) in gates:
            self._check_edge(source, destination)
            wanted.setdefault(destination, []).append(source)
        if not wanted:
            yield
            return

        # The gate values are fixed for the life of this context, so what each
        # one *is* gets decided once here rather than once per edge per forward
        # pass. Read off the device in a single transfer: `float(g)` per gate
        # would be one synchronize per edge, which on a 28-layer model is 14140
        # of them before the first token.
        order = list(gates)
        values = torch.stack([gates[edge].reshape(()) for edge in order]).detach().cpu()
        value_of = dict(zip(order, values.tolist(), strict=True))
        # A gate of exactly 1 contributes `(1 - 1) * live == 0`. Dropping those
        # is not an approximation and it is most of the work: a circuit at 26%
        # density leaves 3720 of 14140 edges open, and each was costing a
        # multiply and a subtract over a full activation tensor to add nothing.
        contributing: Dict[str, List[str]] = {
            destination: [source for source in sources
                          if value_of[(source, destination)] != 1.0]
            for destination, sources in wanted.items()
        }
        # ...and when every contributing gate is shut, the weighted sum is a
        # plain sum, so the multiply goes too. This is the serving case: a
        # circuit is a set of edges, not a set of dials.
        binary = all(value_of[(source, destination)] == 0.0
                     for destination, sources in contributing.items() for source in sources)

        merged: Dict[int, torch.Tensor] = {}
        mlp_out: Dict[int, torch.Tensor] = {}
        embedded: Dict[int, torch.Tensor] = {}
        # One reconstruction per *source* per forward pass, not one per edge. A
        # source's write is a function of the captures alone and those are taken
        # once per pass, so the same tensor was being rebuilt for every
        # destination that reads it -- 14140 rebuilds where 505 will do on a
        # 28-layer model, each holding its own activations for the backward
        # pass. Cleared at the embedding hook, the one thing that fires exactly
        # once at the start of every forward.
        rebuilt: Dict[str, torch.Tensor] = {}
        handles = []
        reconstructing = {"busy": False}

        # Not detached, unlike edge_patch's. These are what the gate's gradient
        # has to flow back through: an edge closed at layer 0 changes the
        # residual at layer 1, which changes what layer 1 writes, and so on.
        def capture_embedding(module, args):
            if not reconstructing["busy"]:
                embedded[0] = args[0]
                rebuilt.clear()

        def capture_merged(index):
            def hook(module, args):
                if not reconstructing["busy"]:
                    merged[index] = args[0]
            return hook

        def capture_mlp(index):
            def hook(module, args, output):
                if not reconstructing["busy"]:
                    mlp_out[index] = output[0] if isinstance(output, tuple) else output
            return hook

        def destination_hook(destination):
            # Two source lists, and which one is used decides nothing about the
            # arithmetic -- only how many kernels it takes. `wanted` keeps every
            # edge because the training path needs a gradient to reach every
            # gate, including the open ones whose contribution is currently
            # zero; `contributing` is the subset that can change the answer.
            every, some = wanted[destination], contributing[destination]

            def reconstruct(sources, residual):
                out = []
                reconstructing["busy"] = True
                try:
                    for source in sources:
                        live = rebuilt.get(source)
                        if live is None:
                            live = self._live_write(source, merged, mlp_out, embedded, residual,
                                                    grad=True)
                            rebuilt[source] = live
                        out.append(live)
                finally:
                    reconstructing["busy"] = False
                return out

            def hook(module, args):
                residual = args[0]
                # The gradient path, unchanged: one term at a time, every edge,
                # so `(1 - g) * live` is a node in the graph for every gate the
                # optimizer has to move. Vectorising this would hold the whole
                # stack of source writes per destination for the backward pass,
                # and a 28-layer model has 56 destinations.
                if torch.is_grad_enabled():
                    delta = torch.zeros_like(residual)
                    for source, live in zip(every, reconstruct(every, residual), strict=True):
                        gate = gates[(source, destination)].to(residual.device)
                        delta = delta - (1.0 - gate).to(residual.dtype) * live
                    return (residual + delta, *args[1:])

                # Inference: the gates are constants, so the sum over sources is
                # one reduction instead of a Python loop of multiply-subtracts.
                # That loop was ~3 kernel launches per edge on every forward --
                # 40k of them per token on a 14140-edge circuit, over activation
                # tensors small enough that the launches, not the arithmetic,
                # were the cost.
                if not some:
                    return args
                lives = reconstruct(some, residual)
                stacked = torch.stack(lives, dim=0)
                if binary:
                    delta = -stacked.sum(dim=0)
                else:
                    weights = torch.tensor(
                        [1.0 - value_of[(source, destination)] for source in some],
                        device=residual.device, dtype=stacked.dtype)
                    delta = -(weights.view(-1, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
                return (residual + delta.to(residual.dtype), *args[1:])
            return hook

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
