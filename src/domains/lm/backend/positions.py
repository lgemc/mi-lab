"""
Which position is the real one, once padding is in the batch.

Prompts are right-padded to one length so a chunk stays comparable, and from
there every read has to say what it means by "the last token" and by "the
average". Both answers are the mask, never the tensor's own shape: reading
column -1 hands back padding for every prompt shorter than the longest in its
batch, and averaging without weights averages padding in. Two functions, both
of them the same correction.

A common pipe could be: _encode | model | _reduce
"""

import torch

from ....core.config import Position


def _last_real(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-wise pick of the final non-padding position from a [batch, seq, ...] tensor

    With right padding the final real token is at mask.sum() - 1, which is the
    last column only for the longest prompt in the batch. Reading column -1
    instead is the bug that makes a short prompt's answer come out of padding.
    """
    index = (mask.sum(dim=1) - 1).long()
    return sequence[torch.arange(sequence.shape[0], device=sequence.device), index]

def _reduce(stacked: torch.Tensor, mask: torch.Tensor, position: Position) -> torch.Tensor:
    """Collapse a [batch, layer, seq, d_model] capture to the requested position

    Padding is never averaged into a MEAN and never mistaken for the LAST
    token: with right padding the final real token is at mask.sum() - 1, which
    is not the final column whenever a prompt is shorter than its batch.
    """
    if position is Position.ALL:
        return stacked
    if position is Position.MEAN:
        weights = mask[:, None, :, None].to(stacked.dtype)
        return (stacked * weights).sum(dim=2) / weights.sum(dim=2)
    last = (mask.sum(dim=1) - 1)[:, None, None, None].expand(-1, stacked.shape[1], -1, stacked.shape[3])
    return stacked.gather(2, last).squeeze(2)
