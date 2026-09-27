"""Adam over a dict of numpy arrays, and a seeded generator every trainer shares

Written out rather than imported because it is twenty lines and the alternative
is torch in an image whose whole job is to answer sliders.
"""

from typing import Dict

import numpy as np


class Adam:
    """Adam (Kingma & Ba, 2015) updating a dict of parameters in place"""

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-2, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def unit(v: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalize along an axis, leaving an all-zero vector at zero rather than NaN"""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0, 1, n)


def sparse_features(generator: np.random.Generator, batch: int, n: int, p_active, low: float = 0.0,
                    high: float = 1.0) -> np.ndarray:
    """Each feature on with probability p (scalar or per-feature), magnitude uniform in [low, high]"""
    on = generator.random((batch, n)) < np.asarray(p_active)
    return on * generator.uniform(low, high, (batch, n))
