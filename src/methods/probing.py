from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..core.metrics import Cost, accuracy, measure, roc_auc
from ..data.dataset import LabeledPrompts

"""
A probe is the cheapest thing that can read a property off a residual stream:
one direction and a bias. That is the point -- if a linear probe finds the
property, the model represents it linearly at that layer, and the artifact
that ships is a few kilobytes of floats rather than a second model.

Probes here are self-contained artifacts. A saved probe carries the layer it
was read from and how deep that layer is, the model it was read from, and the
standardization it was fit with, because a direction without those facts
cannot be applied to anything and cannot be checked by anyone. The depth is
there for the same reason nothing else in the framework names a layer index:
an index means one place in a shallow model and another in a deep one.

difference_of_means is included as the baseline to beat: it needs no training
at all, it is the same vector that steers in adapter.steer, and a trained
probe that does not clear it is not earning its optimizer.

A common pipe could be: capture | train_probe | evaluate | save
"""

class ProbeError(ValueError):
    """Raised when a probe is asked for something it cannot do: wrong width, one-class data, no examples"""

@dataclass
class LinearProbe:
    """One direction, a bias, and everything needed to apply them honestly"""
    weight: torch.Tensor
    bias: float
    mean: torch.Tensor
    std: torch.Tensor
    layer: int
    model_id: str
    n_layers: Optional[int] = None
    position: str = "last"
    dataset: str = "unnamed"
    method: str = "logistic"
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def d_model(self) -> int:
        return int(self.weight.numel())

    @property
    def frac(self) -> Optional[float]:
        """The depth this probe was read from, or None if the model's depth was not recorded

        The layer index alone does not transfer: layer 8 means one place in a
        shallow model and another in a deep one, and the fraction is the half
        of that pair a receiving lab can act on. Older probes predate the
        field and honestly answer None rather than guessing.
        """
        if not self.n_layers:
            return None
        return self.layer / self.n_layers

    @property
    def direction(self) -> torch.Tensor:
        """The probe's direction in activation space, as opposed to standardized space

        `weight` is fit against standardized features, so it is only a
        direction in those coordinates. Expanding the score,
        ((x - mean) / std) @ weight + bias = x @ (weight / std) + constant,
        which makes weight / std the vector that actually points along the
        probe in the residual stream.

        This is the one to steer with. Steering with `weight` directly looks
        almost right and is not: it reweights every dimension by that layer's
        activation scale, which on a residual stream spans orders of
        magnitude.
        """
        return self.weight / self.std

    @property
    def n_bytes(self) -> int:
        """Size of the artifact that would ship, in bytes

        A number worth quoting: a probe is kilobytes where the judge it
        replaces is a network call.
        """
        return sum(tensor.numel() * tensor.element_size() for tensor in (self.weight, self.mean, self.std))

    def score(self, activations: torch.Tensor) -> torch.Tensor:
        """Signed distance from the decision boundary, one score per row

        Positive means the positive class. These are logits, not
        probabilities: rank-based metrics do not care, and a sigmoid would
        only hide how far from the boundary a point sits.
        """
        if activations.dim() == 3 and activations.shape[1] == 1:
            activations = activations[:, 0]
        if activations.dim() != 2:
            raise ProbeError(f"expected [batch, d_model] activations, got shape {tuple(activations.shape)}")
        if activations.shape[1] != self.d_model:
            raise ProbeError(
                f"probe was fit on d_model {self.d_model} but got {activations.shape[1]}; "
                "this probe belongs to a different model"
            )
        standardized = (activations.to(self.weight.dtype) - self.mean) / self.std
        return standardized @ self.weight + self.bias

    def save(self, path: str) -> None:
        """Write the probe and its provenance to one file"""
        payload = asdict(self)
        payload["bias"] = float(self.bias)
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str) -> "LinearProbe":
        """Read a probe back, refusing anything that is not one"""
        source = Path(path)
        if not source.exists():
            raise ProbeError(f"no probe at {source}")
        payload = torch.load(source, weights_only=False)
        allowed_missing = {"metrics", "method", "position", "dataset", "n_layers"}
        missing = set(cls.__dataclass_fields__) - set(payload) - allowed_missing
        if missing:
            raise ProbeError(f"{source} is missing probe fields {sorted(missing)}")
        return cls(**payload)

def _check(activations: torch.Tensor, labels: Sequence[int]) -> torch.Tensor:
    """Validate a training set and return the labels as a float tensor"""
    if activations.dim() == 3 and activations.shape[1] == 1:
        activations = activations[:, 0]
    if activations.dim() != 2:
        raise ProbeError(f"expected [batch, d_model] activations, got shape {tuple(activations.shape)}")
    if activations.shape[0] != len(labels):
        raise ProbeError(f"{activations.shape[0]} activations but {len(labels)} labels")
    label_tensor = torch.as_tensor(labels, dtype=torch.float64)
    if len({int(value) for value in labels}) < 2:
        raise ProbeError("training needs both classes present")
    return label_tensor

def difference_of_means(
    activations: torch.Tensor,
    labels: Sequence[int],
    layer: int,
    model_id: str,
    **provenance,
) -> LinearProbe:
    """The training-free baseline: the direction from the negative mean to the positive mean

    Fit in one pass, no optimizer, no hyperparameters. It is the number a
    trained probe has to beat before its extra machinery is worth anything.
    """
    if activations.dim() == 3 and activations.shape[1] == 1:
        activations = activations[:, 0]
    label_tensor = _check(activations, labels)
    features = activations.to(torch.float64)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    standardized = (features - mean) / std

    positive = standardized[label_tensor == 1].mean(dim=0)
    negative = standardized[label_tensor == 0].mean(dim=0)
    weight = positive - negative
    weight = weight / weight.norm().clamp_min(1e-12)
    # centre the boundary between the two class means along the direction
    bias = -float(((positive + negative) / 2) @ weight)
    return LinearProbe(
        weight=weight, bias=bias, mean=mean, std=std,
        layer=layer, model_id=model_id, method="difference_of_means", **provenance,
    )

def train_probe(
    activations: torch.Tensor,
    labels: Sequence[int],
    layer: int,
    model_id: str,
    epochs: int = 400,
    lr: float = 0.05,
    l2: float = 1e-2,
    seed: int = 0,
    **provenance,
) -> LinearProbe:
    """Fit a logistic probe by full-batch gradient descent

    Features are standardized with statistics taken from *this* data and
    stored on the probe, so the same transform is applied at inference and a
    probe fit at one layer cannot be silently applied at another whose
    activations live on a different scale.

    l2 is not decoration at these sample sizes: with d_model features and a
    few hundred examples the problem is separable, and an unregularized fit
    will happily run the weights off to infinity.
    """
    label_tensor = _check(activations, labels)
    if activations.dim() == 3:
        activations = activations[:, 0]
    features = activations.to(torch.float64)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-6)
    standardized = (features - mean) / std

    torch.manual_seed(seed)
    weight = torch.zeros(standardized.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(standardized @ weight + bias, label_tensor) + l2 * weight.pow(2).sum()
        loss.backward()
        optimizer.step()

    return LinearProbe(
        weight=weight.detach(), bias=float(bias.detach()), mean=mean, std=std,
        layer=layer, model_id=model_id, method="logistic",
        metrics={"train_loss": float(loss.detach())}, **provenance,
    )

def evaluate(probe: LinearProbe, activations: torch.Tensor, labels: Sequence[int]) -> Dict[str, float]:
    """Score held-out activations and report the numbers that decide anything"""
    scores = probe.score(activations)
    return {
        "auc": roc_auc(scores, labels),
        "accuracy": accuracy(scores, labels),
        "n": float(len(labels)),
    }

@dataclass
class LayerReport:
    """What one layer scored, and the probe that scored it"""
    layer: int
    frac: float
    metrics: Dict[str, float]
    probe: LinearProbe

    @property
    def auc(self) -> float:
        return self.metrics["auc"]

def sweep(
    adapter,
    train: LabeledPrompts,
    test: LabeledPrompts,
    fracs: Optional[Sequence[float]] = None,
    method: str = "logistic",
    **train_kwargs,
) -> List[LayerReport]:
    """Train one probe per depth fraction and report where the signal lives

    Activations are captured once for every layer and reused, because the
    forward pass is the expensive part and running it per layer is how a
    sweep turns into an afternoon.

    Which layer wins is a fact about this model; that the winner sits in the
    middle rather than at either end is the part expected to survive a model
    swap.
    """
    fracs = list(fracs) if fracs is not None else [index / 8 for index in range(9)]
    layers = adapter.cfg.layers(fracs)
    train_activations = adapter.capture(train.texts, layers=layers)
    test_activations = adapter.capture(test.texts, layers=layers)

    fit = difference_of_means if method == "difference_of_means" else train_probe
    reports = []
    for position, layer in enumerate(layers):
        probe = fit(
            train_activations[:, position], train.labels, layer=layer,
            model_id=adapter.cfg.id, n_layers=adapter.cfg.n_layers, dataset=train.name, **train_kwargs,
        )
        metrics = evaluate(probe, test_activations[:, position], test.labels)
        probe.metrics.update(metrics)
        reports.append(LayerReport(layer=layer, frac=layer / adapter.cfg.n_layers, metrics=metrics, probe=probe))
    return reports

def measure_scoring_cost(probe: LinearProbe, activations: torch.Tensor, repeats: int = 100) -> Cost:
    """Time the probe alone, with the forward pass already paid for

    This is the marginal cost of the monitoring, which is the number that
    decides whether it can run inline. The forward pass is not counted because
    production is running it anyway.
    """
    with measure(items=activations.shape[0] * repeats) as cost:
        for _ in range(repeats):
            probe.score(activations)
    return cost[0]
