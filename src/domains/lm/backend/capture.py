"""
Reading the residual stream, and adding a direction to it.

Both halves of probing, at one site: the stream leaving a decoder block, read
with a forward hook and written with a forward hook. Capture deliberately does
*not* use output_hidden_states, which looks like the obvious way to do this and
is subtly not the same thing -- that tuple's last entry is the final layer norm
applied to the residual stream rather than the stream, so capturing at depth
1.0 would silently hand back a different quantity than capturing anywhere else,
and Transformers implements it with its own recorder hooks that can observe a
block before an intervention hook is done with it.

Steering is measured in mean activation norms of the layer's own forward pass,
so a strength of 1.0 is the same size of intervention on any model.

A common pipe could be: capture | train_probe | steer
"""

from contextlib import contextmanager
from typing import Iterator, Optional, Sequence

import torch

from ....core.config import ConfigError, Position
from .positions import _reduce


class CaptureMixin:
    """Residual stream reads and residual stream edits, at a block's output"""

    def capture(
        self,
        prompts: Sequence[str],
        layers: Optional[Sequence[int]] = None,
        position: Position = Position.LAST,
    ) -> torch.Tensor:
        """Capture the residual stream at the given layers, batched per cfg.batch_size

        Chunks are accumulated on CPU: on a unified-memory machine the device
        and host pools are the same silicon, but on a discrete GPU holding a
        thousand prompts' activations resident is how capture OOMs.
        """
        if not prompts:
            raise ConfigError("capture needs at least one prompt")
        position = Position(position)
        layers = self._resolve_layers(layers)
        input_ids, attention_mask = self._encode(prompts, padding_side="right")

        chunks = []
        for ids, mask in self._chunks(input_ids, attention_mask):
            with self._record(layers) as captured, torch.no_grad():
                self.model(ids, attention_mask=mask, use_cache=False)
            stacked = torch.stack([captured[index] for index in layers], dim=1)
            chunks.append(_reduce(stacked, mask, position).float().cpu())
        return torch.cat(chunks, dim=0)

    @contextmanager
    def steer(self, layer: int, vector: torch.Tensor, strength: float) -> Iterator[None]:
        """Add a direction to the residual stream leaving `layer`, for the duration of the block

        The vector is normalized and rescaled to the mean activation norm seen
        at that layer in this very forward pass, so a strength of 1.0 means the
        same intervention size on GPT-2 small as on a 27B. At strength 0 no hook
        fires at all, which keeps a zero-strength run byte-identical to no
        steering -- the check that tells you the hook is where you think it is.
        """
        (layer,) = self._resolve_layers([layer])
        if strength == 0:
            yield
            return

        direction = vector.to(self.model.device, self.model.dtype)
        direction = direction / direction.norm()

        def hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            scale = strength * hidden.norm(dim=-1, keepdim=True).mean()
            steered = hidden + scale * direction
            return (steered, *output[1:]) if is_tuple else steered

        handle = self.blocks[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
