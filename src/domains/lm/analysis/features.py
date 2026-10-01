"""Frozen-dictionary diffing: one checkpoint's transcoders read on two checkpoints over the same text

This is the transfer measurement of Kissane et al. (2024, "SAEs (usually)
transfer between base and chat models"), not crosscoder diffing: a dictionary
fitted on one checkpoint cannot contain a feature a fine-tune invented, and
reports it as error. So everything here is phrased as what the *old* basis can
and cannot see of the change, and three numbers carry it:

- **fit**: per layer, the share of the MLP output's variance the dictionary
  misses on each checkpoint, and the paired difference. Pooled over positions
  (sum of residuals over sum of centred variance), never a mean of per-sequence
  ratios, so a short sequence does not count as much as a long one.
- **visible change**: of the change in MLP output between the checkpoints on the
  same position, `dy = y_post - y_pre`, the share the dictionary's own change
  reproduces, `1 - |dy - dyhat|^2 / |dy|^2`. A fit can stay flat while the change
  happens entirely in the residual; this is the number that says so.
- **features**: per-feature firing frequency on both checkpoints, and how many
  features moved by more than a factor of two. The noise floor is like for like:
  the count *across* checkpoints is taken between disjoint halves of the
  sequences (pre on one half, post on the other), and set against the same
  count *within* each checkpoint between the same two halves. Both sides then
  carry the same sampling noise, and the excess is the change. The full-sample
  paired count is reported too, but it has no noise floor of its own. Jaccard
  over the features active at a stated frequency floor, split the same way.
  A feature-overlap number without its granularity and its noise floor is a
  number about superposition (proposal, pitfall 6).

The transcoders are read straight off the release's safetensors and applied by
hand rather than through circuit-tracer's replacement model: two plain
checkpoints share a process this way and TransformerLens never has to be told
about a local fine-tune. Checked against circuit-tracer on one layer of one
fine-tune: fvu 0.579 by hand against 0.573, l0 9.1 against 9.2.

Every per-sequence sum is kept, so the confidence intervals resample sequences
-- the unit that was drawn -- rather than positions, which are not independent.

A common pipe could be: load_transcoders | compare(pre, post, sequences) | FeatureDiff.as_dict
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ....core.config import ConfigError
from ..backend.layout import _blocks, _mlp, _mlp_norm

# Per-sequence sums, in this order, per layer.
SUMS = ("res_pre", "cent_pre", "res_post", "cent_post", "dy", "dy_miss", "y_pre")
# A feature moved if its frequency changed by more than this factor.
SHIFT_FACTOR = 2.0


class FeatureDiffError(ConfigError):
    """A frozen-dictionary diff that cannot be run as asked"""


@dataclass
class LayerTranscoder:
    """One layer's per-layer transcoder: a ReLU encoder over the MLP input, a linear decoder onto its output"""
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x.to(self.W_enc.dtype) @ self.W_enc.T + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


def load_transcoders(release: str, n_layers: int, device: str) -> List[LayerTranscoder]:
    """Every layer of a per-layer transcoder set, refused if it is anything else

    The release says what it is in `config.yaml` (`model_kind`) and how it was
    trained in `wandb-config.yaml` (`act_fn`). This reader implements one shape,
    a ReLU transcoder set, so a cross-layer release or a JumpReLU one is refused
    by name rather than decoded as something it is not.
    """
    import yaml
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = yaml.safe_load(Path(hf_hub_download(release, "config.yaml")).read_text())
    if config.get("model_kind") != "transcoder_set":
        kind = config.get("model_kind")
        raise FeatureDiffError(f"{release} is a '{kind}', and this reads only a per-layer transcoder_set")
    trained = yaml.safe_load(Path(hf_hub_download(release, "wandb-config.yaml")).read_text())
    activation = trained.get("act_fn", {}).get("value")
    if activation != "relu":
        raise FeatureDiffError(f"{release} was trained with act_fn '{activation}', and this applies ReLU")
    layers = []
    for layer in range(n_layers):
        weights = load_file(hf_hub_download(release, f"layer_{layer}.safetensors"), device=device)
        layers.append(LayerTranscoder(weights["W_enc"], weights["b_enc"], weights["W_dec"], weights["b_dec"]))
    return layers


def capture_mlp(model, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every block's MLP input (after its norm) and output for one sequence, as [layers, positions, width]

    The input is read where a transcoder reads it -- the pre-MLP norm's output --
    and the output is the MLP's own return value, before the residual add.
    """
    inputs: Dict[int, torch.Tensor] = {}
    outputs: Dict[int, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(_blocks(model)):
        handles.append(_mlp_norm(block, index).register_forward_hook(
            lambda _m, _i, out, at=index: inputs.__setitem__(at, out[0].detach())))
        handles.append(_mlp(block, index).register_forward_hook(
            lambda _m, _i, out, at=index: outputs.__setitem__(at, out[0].detach())))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for handle in handles:
            handle.remove()
    order = sorted(inputs)
    return torch.stack([inputs[i] for i in order]), torch.stack([outputs[i] for i in order])


@dataclass
class FeatureDiff:
    """Everything one pair produced over one text set, with the per-sequence sums kept for resampling"""
    text_set: str
    layers: int
    features: int
    sums: List[List[List[float]]] = field(default_factory=list)   # [sequence][layer][SUMS]
    positions: int = 0
    # Firing counts, [model][half][layer][feature]; halves split sequences even/odd.
    counts: Optional[torch.Tensor] = None
    halves: List[int] = field(default_factory=lambda: [0, 0])

    def add(self, per_layer: List[List[float]], positions: int, half: int) -> None:
        self.sums.append(per_layer)
        self.positions += positions
        self.halves[half] += positions

    # --- fit and visible change, with bootstrap intervals over sequences

    def _pooled(self, rows: Sequence[int]) -> Dict[str, List[float]]:
        if getattr(self, "_array", None) is None or self._array.shape[0] != len(self.sums):
            self._array = torch.tensor(self.sums, dtype=torch.float64)            # [sequence, layer, sums]
        totals = self._array[torch.as_tensor(list(rows))].sum(0)                  # [layer, sums]
        column = {name: totals[:, i] for i, name in enumerate(SUMS)}
        fvu_pre = column["res_pre"] / column["cent_pre"]
        fvu_post = column["res_post"] / column["cent_post"]
        return {
            "fvu_pre": fvu_pre.tolist(),
            "fvu_post": fvu_post.tolist(),
            "fvu_delta": (fvu_post - fvu_pre).tolist(),
            "visible_change": (1 - column["dy_miss"] / column["dy"]).tolist(),
            "change_size": (column["dy"] / column["y_pre"]).tolist(),
        }

    def fit(self, resamples: int = 1000, seed: int = 0) -> Dict[str, object]:
        """Point estimates per layer and for the layer mean, each with a 95% bootstrap interval"""
        everything = range(len(self.sums))
        point = self._pooled(everything)
        draws = {name: [] for name in point}
        means = {name: [] for name in point}
        generator = random.Random(seed)
        for _ in range(resamples):
            rows = [generator.randrange(len(self.sums)) for _ in everything]
            sample = self._pooled(rows)
            for name, values in sample.items():
                draws[name].append(values)
                means[name].append(sum(values) / len(values))

        def interval(values: List[float]) -> List[float]:
            ordered = sorted(values)
            return [ordered[int(0.025 * len(ordered))], ordered[int(0.975 * len(ordered)) - 1]]

        report = {}
        for name, values in point.items():
            per_layer = [interval([draw[layer] for draw in draws[name]]) for layer in range(self.layers)]
            report[name] = {
                "per_layer": values,
                "per_layer_ci95": per_layer,
                "mean": sum(values) / len(values),
                "mean_ci95": interval(means[name]),
            }
        return report

    # --- features

    def feature_report(self, floor: float, top: int = 20) -> Dict[str, object]:
        """Frequency shifts and active-set overlap per layer, each beside its split-half noise floor

        `floor` is the firing frequency (share of positions) above which a feature
        counts as active; it is the granularity the Jaccard is stated at.
        """
        counts = self.counts.double().cpu()                                    # [model, half, layer, feature]
        halves = torch.tensor(self.halves, dtype=torch.float64).view(1, 2, 1, 1)
        frequency_half = counts / halves.clamp(min=1)
        frequency = counts.sum(1) / max(self.positions, 1)                    # [model, layer, feature]
        epsilon = 1.0 / max(self.positions, 1)

        def moved(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            ratio = torch.log2((second + epsilon) / (first + epsilon))
            busy = torch.maximum(first, second) >= floor
            return (ratio.abs() > torch.log2(torch.tensor(SHIFT_FACTOR))) & busy

        def jaccard(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            a, b = first >= floor, second >= floor
            union = (a | b).sum(-1).double()
            return torch.where(union > 0, (a & b).sum(-1).double() / union, torch.full_like(union, float("nan")))

        paired = moved(frequency[0], frequency[1]).sum(-1)
        # Like for like: both counts compare two half-size samples drawn from disjoint sequences.
        across = (moved(frequency_half[0, 0], frequency_half[1, 1]).sum(-1)
                  + moved(frequency_half[0, 1], frequency_half[1, 0]).sum(-1)) / 2
        within = (moved(frequency_half[0, 0], frequency_half[0, 1]).sum(-1)
                  + moved(frequency_half[1, 0], frequency_half[1, 1]).sum(-1)) / 2
        jaccard_across = (jaccard(frequency_half[0, 0], frequency_half[1, 1])
                          + jaccard(frequency_half[0, 1], frequency_half[1, 0])) / 2
        jaccard_within = (jaccard(frequency_half[0, 0], frequency_half[0, 1])
                          + jaccard(frequency_half[1, 0], frequency_half[1, 1])) / 2
        shift = torch.log2((frequency[1] + epsilon) / (frequency[0] + epsilon))
        shift = torch.where(torch.maximum(frequency[0], frequency[1]) >= floor, shift, torch.zeros_like(shift))
        layers = []
        for layer in range(self.layers):
            order = shift[layer].abs().argsort(descending=True)[:top]
            layers.append({
                "layer": layer,
                "active_pre": int((frequency[0, layer] >= floor).sum()),
                "active_post": int((frequency[1, layer] >= floor).sum()),
                "l0_pre": float(frequency[0, layer].sum()),
                "l0_post": float(frequency[1, layer].sum()),
                "moved_paired": int(paired[layer]),
                "moved_across_halves": float(across[layer]),
                "moved_within_halves": float(within[layer]),
                "jaccard_paired": float(jaccard(frequency[0, layer], frequency[1, layer])),
                "jaccard_across_halves": float(jaccard_across[layer]),
                "jaccard_within_halves": float(jaccard_within[layer]),
                "top_shifted": [
                    {"feature": int(j), "freq_pre": float(frequency[0, layer, j]),
                     "freq_post": float(frequency[1, layer, j]), "log2_ratio": float(shift[layer, j])}
                    for j in order
                ],
            })
        return {"floor": floor, "shift_factor": SHIFT_FACTOR, "positions": self.positions, "layers": layers}


def compare(pre, post, transcoders: List[LayerTranscoder], sequences: Sequence[Tuple[torch.Tensor, int]],
            text_set: str) -> FeatureDiff:
    """One dictionary over two checkpoints on identical sequences

    Each sequence comes with the first position to score: 1 for prose (position
    0 carries the attention sink's outsized norm, and circuit-tracer zeroes its
    features), the first answer position for a templated task, so that what is
    measured is the part a fine-tune was trained on.
    """
    width = transcoders[0].W_enc.shape[0]
    diff = FeatureDiff(text_set=text_set, layers=len(transcoders), features=width)
    device = transcoders[0].W_enc.device
    diff.counts = torch.zeros(2, 2, len(transcoders), width, device=device)
    for index, (ids, start) in enumerate(sequences):
        half = index % 2
        in_pre, out_pre = capture_mlp(pre, ids)
        in_post, out_post = capture_mlp(post, ids)
        per_layer = []
        with torch.no_grad():
            for layer, transcoder in enumerate(transcoders):
                x_pre, y_pre = in_pre[layer, start:], out_pre[layer, start:].float()
                x_post, y_post = in_post[layer, start:], out_post[layer, start:].float()
                f_pre, f_post = transcoder.encode(x_pre), transcoder.encode(x_post)
                yhat_pre, yhat_post = transcoder.decode(f_pre).float(), transcoder.decode(f_post).float()
                diff.counts[0, half, layer] += (f_pre > 0).sum(0)
                diff.counts[1, half, layer] += (f_post > 0).sum(0)
                dy = y_post - y_pre
                per_layer.append([
                    ((y_pre - yhat_pre) ** 2).sum().item(),
                    ((y_pre - y_pre.mean(0, keepdim=True)) ** 2).sum().item(),
                    ((y_post - yhat_post) ** 2).sum().item(),
                    ((y_post - y_post.mean(0, keepdim=True)) ** 2).sum().item(),
                    (dy ** 2).sum().item(),
                    ((dy - (yhat_post - yhat_pre)) ** 2).sum().item(),
                    (y_pre ** 2).sum().item(),
                ])
        diff.add(per_layer, int(ids.shape[1] - start), half)
    return diff
