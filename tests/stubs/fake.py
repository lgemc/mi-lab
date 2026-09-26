from contextlib import contextmanager
from typing import List, Sequence

import torch

from src.core.config import ModelConfig

"""
A model and a task with no weights and no tokenizer behind them.

The measurement contract's claim is that `methods/` consumes a batch of inputs,
a set of sites and one scalar per example, and knows nothing else. A claim like
that is testable exactly once you can hand the layer something that is not a
language model at all -- so these satisfy CircuitAdapter and CircuitTask while
being arithmetic over integers, and a test that passes with them is a test that
did not reach for a tokenizer.

They are deliberately not a mock: nothing here records calls or asserts. The
`outputs` of FakeAdapter is a real function of its inputs, so a baseline
measured on `clean` and one measured on `corrupted` genuinely differ and the
span is a real number.
"""


def fake_config(**overrides) -> ModelConfig:
    """A config for a model that does not exist, with the sizes already stamped in"""
    fields = {"id": "fake", "backend": "none", "hf_name": "fake", "n_layers": 4, "d_model": 8, "n_heads": 2}
    fields.update(overrides)
    return ModelConfig(**fields)


class Sum:
    """A Score with no per-example state: the mean of whatever came out

    Well-formed and differentiable, and it names no vocabulary. `select` returns
    self, which is what a score whose definition does not vary by example is.
    """

    name = "sum"
    units = "arbitrary"
    definition = "the mean of the model's output over its last axis, for testing"
    differentiable = True

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        return output.to(torch.float64).mean(dim=-1)

    def select(self, rows: slice) -> "Sum":
        return self


class NoGradient(Sum):
    """The same score, declaring itself undifferentiable: BLEU's shape without BLEU"""

    name = "no-gradient"
    definition = "a score with no derivative, for testing that a gradient technique refuses it"
    differentiable = False


class FakeAdapter:
    """Everything CircuitAdapter asks for, over integers

    `outputs` is `len(input) * scale` per row, so a corrupted batch of shorter
    strings scores lower and the span is real. Nothing here tokenizes: the
    inputs are opaque and only their length is read, which is the point.
    """

    def __init__(self, cfg: ModelConfig = None, scale: float = 1.0):
        self.cfg = cfg if cfg is not None else fake_config()
        self.scale = scale

    def layer(self, frac=None) -> int:
        return self.cfg.layer(frac)

    def outputs(self, inputs: Sequence) -> torch.Tensor:
        rows = [[float(len(str(item))) * self.scale] * self.cfg.d_model for item in inputs]
        return torch.tensor(rows, dtype=torch.float32)

    def capture(self, prompts, layers=None, position=None) -> torch.Tensor:
        return torch.zeros(len(list(prompts)), self.cfg.n_layers, self.cfg.d_model)

    def generate(self, prompts, max_new_tokens=None, **kwargs) -> List[str]:
        return ["" for _ in prompts]

    def attention(self, prompts, layers=None) -> torch.Tensor:
        return torch.zeros(len(list(prompts)), self.cfg.n_layers, self.cfg.n_heads, 1, 1)

    def head_outputs(self, prompts, layers=None) -> torch.Tensor:
        return torch.zeros(len(list(prompts)), self.cfg.n_layers, self.cfg.n_heads, 1, 1)

    def gradients(self, inputs, readout, layers=None, toward=None, alpha=1.0) -> torch.Tensor:
        return torch.zeros(len(list(inputs)), self.cfg.n_layers, self.cfg.n_heads, 1, 1)

    def decompose(self, prompts):
        raise NotImplementedError("the fake model has no residual stream to split")

    @contextmanager
    def patch(self, residual=None, heads=None):
        yield

    @contextmanager
    def steer(self, layer, vector, strength):
        yield


class FakeTask:
    """A CircuitTask whose inputs are opaque and whose readout names no tokens"""

    name = "fake"
    modality = "none"

    def __init__(self, score=None, size: int = 4):
        # the corrupted twin is one character shorter, which is the whole corruption
        self.clean = [f"clean-{index:02d}" for index in range(size)]
        self.corrupted = [f"corrupt{index:02d}" for index in range(size)]
        self.score = score if score is not None else Sum()

    def __len__(self) -> int:
        return len(self.clean)

    def readout(self, adapter):
        return self.score

    def labels(self, adapter) -> List[str]:
        return ["p0"]

    def landmarks(self, adapter) -> dict:
        return {"END": 0}

    def subset(self, indices) -> "FakeTask":
        chosen = FakeTask(score=self.score, size=0)
        chosen.clean = [self.clean[index] for index in indices]
        chosen.corrupted = [self.corrupted[index] for index in indices]
        return chosen
