"""What changed between two checkpoints, starting with the part that needs no dictionary.

`~/main/m/proposals/model-diffing-sdft-transcoders.md` asks what a fine-tune does
to a model's insides and lists four ways to answer it. This module is the first,
and first because it is free: the difference between two weight matrices is
arithmetic over two files, with no forward pass, no dictionary, and no training
run to wait for. Zhong & Raghunathan (2508.00161) report that the leading
singular directions of that difference correspond to newly acquired behaviour,
which makes the weight diff the baseline every dictionary-based strategy has to
beat before its cost is justified.

Three numbers per weight tensor, each answering a different question:

  delta_relative   how far the tensor moved, as a fraction of where it was.
                   Scale-free, so an attention projection and an embedding are
                   comparable.
  stable_rank      how many directions the movement used: ||D||_F^2 / ||D||_2^2.
                   One dominant direction gives 1, an isotropic change gives
                   min(shape). The same quantity the plasticity literature
                   tracks (Kumar et al. 2021, Lyle et al. 2023), measured here
                   on the update rather than on an activation covariance.
  top_k_energy     the fraction of ||D||_F^2 in the leading TOP_K directions --
                   the *Watch the Weights* premise as a number rather than an
                   assumption.

Reading them together is the point. A tensor that moved a tenth of its norm in
eight directions and one that moved a tenth over four hundred are different
events, and `delta_relative` alone cannot tell them apart.

Unlike `comparison.py`, which scores results other code produced, this module
opens checkpoint files itself, and it is the only module in `methods/` that
does. Streaming one tensor at a time is what keeps a diff of two mid-sized
checkpoints inside a few GiB; hoisting the reading into a caller would mean
handing this code two whole state dicts, which is the thing being avoided.
`index` decides where each tensor lives, `read` fetches one, and every
measurement below them takes tensors and knows nothing about files.

Singular values are exact by default. One projection-sized `svdvals` is about a
second on CPU and a fraction of that on an accelerator, so a whole pair of
1.7B checkpoints is a minute or two -- cheap enough that an estimate would be a
false economy. `estimate=True` switches to `torch.svd_lowrank`, which agreed to
0.3% on the tensor it was checked against and runs ~25x faster; it exists for
the 8B and above, and every number it produces is stamped `estimated`.

A common pipe could be: index | checkpoint_delta | DeltaReport.profile | table
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from safetensors import safe_open

from ..core.metrics import MetricError

# The leading directions `top_k_energy` sums over. Eight because that is the
# order of "a handful of interpretable directions" the weight-diffing claim is
# about. Nothing downstream depends on the choice: `rank_for_half` reports the
# same curve without a k fixed in advance.
TOP_K = 8

# `model.layers.<i>.<block>.<kind>.weight` -- the only keys carrying both a
# layer index and a module kind, which is every tensor an aggregate groups by.
BLOCK = re.compile(r"^model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight$")

ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
FEEDFORWARD = ("gate_proj", "up_proj", "down_proj")


class DiffingError(MetricError):
    """Two checkpoints that cannot be diffed, said with the tensor that said so"""


# ------------------------------------------------------------- what came back

@dataclass
class TensorDelta:
    """How one weight tensor moved between two checkpoints

    `stable_rank`, `top_k_energy` and `rank_for_half` stay None for anything
    that is not a matrix. A norm's gain vector has no singular values, and
    reporting 1.0 for it would read as "one dominant direction" rather than
    "the question does not apply here".
    """
    name: str
    shape: Tuple[int, ...]
    delta_relative: float
    layer: Optional[int] = None
    kind: Optional[str] = None
    stable_rank: Optional[float] = None
    top_k_energy: Optional[float] = None
    rank_for_half: Optional[int] = None
    estimated: bool = False

    @property
    def full_rank(self) -> Optional[int]:
        """The largest stable rank this tensor's shape allows"""
        return min(self.shape) if len(self.shape) == 2 else None

    @property
    def concentrated(self) -> Optional[float]:
        """Stable rank as a fraction of the rank available, so shapes compare"""
        if self.stable_rank is None or not self.full_rank:
            return None
        return self.stable_rank / self.full_rank

    def __str__(self) -> str:
        head = f"{self.name:<44} moved {self.delta_relative:.1%}"
        if self.stable_rank is None:
            return head
        mark = "~" if self.estimated else ""
        return (
            f"{head} over {mark}{self.stable_rank:>7.1f} of {self.full_rank} directions, "
            f"top{TOP_K} {self.top_k_energy:.1%}"
        )

@dataclass
class DeltaReport:
    """Every shared tensor's movement, and every tensor that was not shared

    `skipped` is not an error log. A checkpoint that ships a tensor its own
    config ties away is ordinary, and a diff that dropped it silently would be
    reporting on a different pair of models than the one it was handed.
    """
    pre: str
    post: str
    deltas: List[TensorDelta] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)

    @property
    def matrices(self) -> List[TensorDelta]:
        """The tensors that have singular values, which is where the rank claims live"""
        return [delta for delta in self.deltas if delta.stable_rank is not None]

    @property
    def layers(self) -> List[int]:
        return sorted({delta.layer for delta in self.deltas if delta.layer is not None})

    @property
    def estimated(self) -> bool:
        return any(delta.estimated for delta in self.deltas)

    def by_kind(self, kind: str) -> List[TensorDelta]:
        return [delta for delta in self.deltas if delta.kind == kind]

    def by_layer(self, layer: int, kinds: Optional[Tuple[str, ...]] = None) -> List[TensorDelta]:
        chosen = [delta for delta in self.deltas if delta.layer == layer]
        return [delta for delta in chosen if delta.kind in kinds] if kinds else chosen

    def profile(self, kinds: Tuple[str, ...]) -> Dict[int, Dict[str, float]]:
        """Mean movement and mean stable rank per layer, over one group of modules

        `profile(ATTENTION)` beside `profile(FEEDFORWARD)` is the layer-wise
        comparison the proposal's fourth experiment asks for: whether a method
        rewrites attention or the feed-forward blocks, and at what depth.
        """
        out: Dict[int, Dict[str, float]] = {}
        for layer in self.layers:
            chosen = [delta for delta in self.by_layer(layer, kinds) if delta.stable_rank is not None]
            if not chosen:
                continue
            count = len(chosen)
            out[layer] = {
                "delta_relative": sum(delta.delta_relative for delta in chosen) / count,
                "stable_rank": sum(delta.stable_rank for delta in chosen) / count,
                "top_k_energy": sum(delta.top_k_energy for delta in chosen) / count,
                "n": count,
            }
        return out

    def as_dict(self) -> Dict[str, object]:
        """The artifact form: flat rows, readable on a machine with no torch"""
        return {
            "pre": self.pre,
            "post": self.post,
            "top_k": TOP_K,
            "estimated": self.estimated,
            "n_tensors": len(self.deltas),
            "skipped": self.skipped,
            "deltas": [
                {
                    "tensor": delta.name,
                    "shape": list(delta.shape),
                    "layer": delta.layer,
                    "kind": delta.kind,
                    "delta_relative": delta.delta_relative,
                    "stable_rank": delta.stable_rank,
                    "full_rank": delta.full_rank,
                    "top_k_energy": delta.top_k_energy,
                    "rank_for_half": delta.rank_for_half,
                }
                for delta in self.deltas
            ],
            "profiles": {
                "attention": {str(layer): row for layer, row in self.profile(ATTENTION).items()},
                "feedforward": {str(layer): row for layer, row in self.profile(FEEDFORWARD).items()},
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "DeltaReport":
        """The report back from its artifact, so a summary re-reads rather than recomputes

        The profiles in the artifact are derived and are dropped here; they are
        recomputed from the rows, which is the only way the two can never
        disagree.
        """
        report = cls(pre=str(payload["pre"]), post=str(payload["post"]), skipped=dict(payload.get("skipped", {})))
        estimated = bool(payload.get("estimated", False))
        for row in payload.get("deltas", []):
            report.deltas.append(
                TensorDelta(
                    name=row["tensor"],
                    shape=tuple(row["shape"]),
                    delta_relative=row["delta_relative"],
                    layer=row["layer"],
                    kind=row["kind"],
                    stable_rank=row["stable_rank"],
                    top_k_energy=row["top_k_energy"],
                    rank_for_half=row["rank_for_half"],
                    estimated=estimated and row["stable_rank"] is not None,
                )
            )
        return report

    def __str__(self) -> str:
        if not self.matrices:
            return f"{self.pre} -> {self.post}: no matrix was shared by both checkpoints"
        count = len(self.matrices)
        moved = sum(delta.delta_relative for delta in self.matrices) / count
        rank = sum(delta.stable_rank for delta in self.matrices) / count
        return f"{self.pre} -> {self.post}: {count} matrices moved {moved:.1%} over {rank:.0f} directions on average"

# ------------------------------------------------------- where the weights are

def resolve(reference: str) -> Path:
    """A local checkpoint directory, from a path or a hub name

    Both forms are load-bearing. A fine-tune arrives as a directory of weights
    -- `configs/qwen3-0.6b-sdft.yaml` says to point `hf_name` at one -- while
    the checkpoint it is diffed against is usually a hub name already in the
    cache. Resolving a name goes through the cache and downloads only the
    weights and configs, never the tokenizer files a diff will not read.
    """
    local = Path(reference).expanduser()
    if local.is_dir():
        return local

    from huggingface_hub import snapshot_download  # not needed for a local fine-tune

    return Path(snapshot_download(reference, allow_patterns=["*.safetensors", "*.json"]))

def index(checkpoint: Path) -> Dict[str, Path]:
    """Which file holds each tensor, for a single-file or a sharded checkpoint

    A map rather than the weights: the memory budget of this module is that a
    tensor is opened when it is wanted and dropped afterwards, so what a caller
    needs is where to look.
    """
    files = sorted(Path(checkpoint).glob("*.safetensors"))
    if not files:
        raise DiffingError(
            f"{checkpoint} holds no .safetensors file, so there is nothing to diff. "
            "Point this at a checkpoint directory or a hub name, not at a config"
        )
    where: Dict[str, Path] = {}
    for path in files:
        with safe_open(path, framework="pt") as handle:
            # Bound first because a safe_open handle is not a mapping: `keys()`
            # is its only listing and iterating the handle itself does nothing.
            keys = handle.keys()
            for key in keys:
                where[key] = path
    return where

def shapes(where: Dict[str, Path]) -> Dict[str, Tuple[int, ...]]:
    """Every tensor's shape, without reading a tensor

    What a dry run needs: which of these are matrices, and therefore how many
    decompositions the real pass will take. safetensors answers that from the
    header, so this is a few file opens rather than a checkpoint's worth of
    reading.
    """
    found: Dict[str, Tuple[int, ...]] = {}
    for path in sorted(set(where.values())):
        with safe_open(path, framework="pt") as handle:
            keys = handle.keys()
            for key in keys:
                found[key] = tuple(handle.get_slice(key).get_shape())
    return found

def read(where: Dict[str, Path], name: str, device: str = "cpu") -> torch.Tensor:
    """One tensor as float32, on `device`

    float32 because every measurement here is taken on a *difference*.
    Subtracting two bfloat16 tensors that agree to three decimals leaves a
    result whose leading bits are all there is, and bfloat16 keeps eight of
    them -- so the norms and singular values of the remainder would be
    quantisation, reported to three significant figures.
    """
    with safe_open(where[name], framework="pt") as handle:
        return handle.get_tensor(name).to(device=device, dtype=torch.float32)

# ------------------------------------------------------------ what moved where

def classify(name: str) -> Tuple[Optional[int], Optional[str]]:
    """A tensor's layer index and module kind, or (None, None) off the stack"""
    found = BLOCK.match(name)
    if not found:
        return None, None
    return int(found.group(1)), found.group(2)

def tensor_delta(name: str, pre: torch.Tensor, post: torch.Tensor, estimate: bool = False) -> TensorDelta:
    """The three numbers, for one pair of tensors"""
    if pre.shape != post.shape:
        raise DiffingError(
            f"{name} is {tuple(pre.shape)} in one checkpoint and {tuple(post.shape)} in the other; "
            "these are not two versions of one model"
        )

    delta = post - pre
    frobenius = delta.norm().item()
    reference = pre.norm().item()
    layer, kind = classify(name)
    relative = frobenius / reference if reference else 0.0

    if delta.ndim != 2 or frobenius == 0.0:
        return TensorDelta(name=name, shape=tuple(pre.shape), delta_relative=relative, layer=layer, kind=kind)

    if estimate:
        # q above TOP_K on purpose: a randomized range finder's last columns are
        # its worst, so the standard fix is to oversample and then discard.
        _, values, _ = torch.svd_lowrank(delta, q=min(TOP_K * 4, *delta.shape), niter=4)
    else:
        values = torch.linalg.svdvals(delta)

    energy = frobenius**2
    cumulative = torch.cumsum(values**2, dim=0) / energy
    reached_half = bool(cumulative[-1] >= 0.5)
    return TensorDelta(
        name=name,
        shape=tuple(pre.shape),
        delta_relative=relative,
        layer=layer,
        kind=kind,
        stable_rank=energy / (values[0].item() ** 2),
        top_k_energy=(values[:TOP_K] ** 2).sum().item() / energy,
        # None when an estimate's directions never reached half the energy,
        # which is a fact about the estimate and not about the tensor.
        rank_for_half=int((cumulative < 0.5).sum().item()) + 1 if reached_half else None,
        estimated=estimate,
    )

def duplicates(where: Dict[str, Path], names: List[str], device: str = "cpu") -> Dict[str, str]:
    """Which of `names` are byte-identical copies of another tensor beside them

    Qwen3-1.7B ships an `lm_head.weight` while its own config sets
    `tie_word_embeddings: true`, and that tensor is an exact copy of the input
    embedding. Counting it would double one tensor's weight in every aggregate,
    and dropping it quietly would leave a diff claiming to have read a
    checkpoint it did not. So it is found, named, and reported as skipped.
    """
    found: Dict[str, str] = {}
    for name in names:
        candidate = read(where, name, device=device)
        for other in where:
            if other == name or other in found:
                continue
            twin = read(where, other, device=device)
            if twin.shape == candidate.shape and torch.equal(twin, candidate):
                found[name] = other
                break
    return found

def checkpoint_delta(
    pre: Path,
    post: Path,
    device: str = "cpu",
    estimate: bool = False,
    on_tensor: Optional[Callable[[TensorDelta], None]] = None,
    labels: Optional[Tuple[str, str]] = None,
) -> DeltaReport:
    """Every shared tensor's movement between two checkpoints

    `on_tensor` fires as each tensor finishes, because a 1.7B pair is a few
    hundred tensors and a minute of silence, and a larger pair is longer than
    that.

    `labels` names the two sides. Without it the report is headed by the
    directory names, and for a checkpoint resolved out of the hub cache that is
    a commit hash -- exact provenance and unreadable, so a caller that knows the
    config ids should say them.
    """
    before, after = index(pre), index(post)
    names = labels or (Path(pre).name, Path(post).name)
    report = DeltaReport(pre=names[0], post=names[1])

    for name in sorted(set(before) - set(after)):
        report.skipped[name] = "present in pre, absent in post"
    only_post = sorted(set(after) - set(before))
    for name, twin in duplicates(after, only_post, device=device).items():
        report.skipped[name] = f"post only, and byte-identical to {twin}, which is counted instead"
    for name in only_post:
        report.skipped.setdefault(name, "present in post, absent in pre")

    for name in sorted(set(before) & set(after)):
        delta = tensor_delta(
            name,
            read(before, name, device=device),
            read(after, name, device=device),
            estimate=estimate,
        )
        report.deltas.append(delta)
        if on_tensor is not None:
            on_tensor(delta)
    return report

# ------------------------------------------------------------- how it is read

def table(report: DeltaReport) -> Iterator[str]:
    """The summary as lines: one per module kind, then one per layer

    Two scripts and the CLI want the same table, so it lives here rather than in
    either of them.
    """
    mark = "~" if report.estimated else ""
    yield f"{'kind':<14}{'n':>4}{'rel move':>10}{'stable rank':>13}{'of':>7}{f'top{TOP_K}':>8}{'half at':>9}"
    for kind in sorted({delta.kind for delta in report.matrices if delta.kind}):
        chosen = [delta for delta in report.by_kind(kind) if delta.stable_rank is not None]
        count = len(chosen)
        half = [delta.rank_for_half for delta in chosen if delta.rank_for_half is not None]
        yield (
            f"{kind:<14}{count:>4}"
            f"{sum(delta.delta_relative for delta in chosen) / count:>10.3f}"
            f"{mark}{sum(delta.stable_rank for delta in chosen) / count:>12.1f}"
            f"{chosen[0].full_rank:>7}"
            f"{sum(delta.top_k_energy for delta in chosen) / count:>8.3f}"
            f"{(sum(half) / len(half)) if half else float('nan'):>9.0f}"
        )

    attention, feedforward = report.profile(ATTENTION), report.profile(FEEDFORWARD)
    if not attention or not feedforward:
        return
    yield ""
    yield f"{'layer':>6}{'attn move':>11}{'ffn move':>10}{'attn srank':>12}{'ffn srank':>11}"
    for layer in report.layers:
        if layer not in attention or layer not in feedforward:
            continue
        heads, blocks = attention[layer], feedforward[layer]
        yield (
            f"{layer:>6}{heads['delta_relative']:>11.3f}{blocks['delta_relative']:>10.3f}"
            f"{heads['stable_rank']:>12.1f}{blocks['stable_rank']:>11.1f}"
        )
