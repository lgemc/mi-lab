"""A probe that is 99% accurate and a steering vector that does nothing

A "model" holds a concept c (say, the sentiment of a review) in its residual
stream twice: once in the direction its downstream computation actually reads
(d_used), and once in a direction nothing reads (d_spurious) that happens to
correlate with c -- think exclamation marks, which travel with positive reviews
without being what the model's verdict depends on.

A linear probe asks "where is c decodable?" and prefers whichever direction is
cleaner, which can be the one the model ignores. Steering asks "what happens if
I push along this direction?", which is the causal question, and only d_used
moves the output. The page lets the correlate get louder and watches probe
accuracy go up while the probe direction's causal effect goes to zero.
"""

from functools import lru_cache
from typing import Dict

import numpy as np

from .optim import rng

D = 6


def directions():
    e = np.eye(D)
    return e[0], (e[1] + 0.3 * e[2]) / np.linalg.norm(e[1] + 0.3 * e[2])


def data(g, n: int, correlation: float, spurious_scale: float, noise: float):
    d_used, d_spur = directions()
    c = g.integers(0, 2, n)
    # the correlate agrees with c with probability (1 + correlation) / 2
    agree = g.random(n) < (1 + correlation) / 2
    s = np.where(agree, c, 1 - c)
    h = (np.outer(c * 1.0 - 0.5, d_used) * 1.0
         + np.outer(s * 1.0 - 0.5, d_spur) * spurious_scale
         + g.normal(scale=noise, size=(n, D))
         + np.outer(g.normal(scale=noise * 1.5, size=n), d_used))  # the used direction is noisier
    return h, c


def model_output(h: np.ndarray) -> np.ndarray:
    """The downstream computation: P(positive) read off d_used only"""
    d_used, _ = directions()
    return 1 / (1 + np.exp(-6 * (h @ d_used)))


def fit_probe(h: np.ndarray, c: np.ndarray, l2: float = 1e-2, steps: int = 400) -> np.ndarray:
    """Logistic regression by gradient descent, returning weights (bias last)"""
    x = np.hstack([h, np.ones((len(h), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w)))
        w -= 0.5 * (x.T @ (p - c) / len(c) + l2 * np.r_[w[:-1], 0])
    return w


@lru_cache(maxsize=128)
def study(correlation: float = 0.9, spurious_scale: float = 1.5, noise: float = 0.3, strength: float = 1.0) -> Dict:
    g = rng(5)
    h, c = data(g, 2000, correlation, spurious_scale, noise)
    ht, ct = data(g, 2000, correlation, spurious_scale, noise)
    w = fit_probe(h, c)
    probe_dir = w[:-1] / np.linalg.norm(w[:-1])
    acc = float((((np.hstack([ht, np.ones((len(ht), 1))]) @ w) > 0) == ct).mean())
    d_used, d_spur = directions()
    dom = ht[ct == 1].mean(0) - ht[ct == 0].mean(0)
    dom /= np.linalg.norm(dom)
    random_dir = rng(9).normal(size=D)
    random_dir /= np.linalg.norm(random_dir)
    negatives = ht[ct == 0]
    base = float(model_output(negatives).mean())
    rows = []
    for name, d in [("probe", probe_dir), ("difference of means", dom), ("true (used) direction", d_used),
                    ("spurious direction", d_spur), ("random", random_dir)]:
        steered = float(model_output(negatives + strength * d).mean())
        rows.append({"name": name, "direction": d.round(3).tolist(), "cos_used": float(d @ d_used),
                     "cos_spurious": float(d @ d_spur), "effect": steered - base})
    # accuracy of a probe restricted to each single direction, for the "where is it decodable" view
    per_dir = {}
    for name, d in [("used", d_used), ("spurious", d_spur)]:
        proj = (h @ d)[:, None]
        wp = fit_probe(proj, c)
        per_dir[name] = float(((((ht @ d) * wp[0] + wp[1]) > 0) == ct).mean())
    shown = 300
    return {
        "probe_accuracy": acc,
        "single_direction_accuracy": per_dir,
        "baseline_output": base,
        "steering": rows,
        "points": np.stack([ht[:shown] @ d_used, ht[:shown] @ d_spur], axis=1).round(3).tolist(),
        "labels": ct[:shown].tolist(),
        "probe_2d": [float(probe_dir @ d_used), float(probe_dir @ d_spur)],
    }
