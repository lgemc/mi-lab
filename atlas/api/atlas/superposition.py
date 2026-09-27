"""Toy Models of Superposition (Elhage et al., 2022), trained in a request

n sparse features squeezed through m < n dimensions and read back out:

    h = W x        (m x n)
    x' = ReLU(W^T h + b)
    loss = sum_i I_i (x_i - x'_i)^2,   I_i = decay^i

Dense features get the m most important features and drop the rest, as PCA
would. Sparse features get *all* of them, at angles, because two features that
are rarely on together interfere rarely -- and the ReLU with a negative bias
filters the small interference out. The page drags the sparsity slider and
watches a pentagon form out of a pair of axes.
"""

from functools import lru_cache
from typing import Dict

import numpy as np

from .optim import Adam, rng, sparse_features


@lru_cache(maxsize=256)
def train(n: int = 5, m: int = 2, sparsity: float = 0.9, decay: float = 0.9, steps: int = 3000,
          seed: int = 0) -> Dict:
    g = rng(seed)
    params = {"W": g.normal(scale=0.3, size=(m, n)), "b": np.zeros(n)}
    opt = Adam(params, lr=1e-2)
    importance = decay ** np.arange(n)
    history = []
    for step in range(steps):
        x = sparse_features(g, 1024, n, 1 - sparsity)
        w, b = params["W"], params["b"]
        h = x @ w.T
        z = h @ w + b
        xh = np.maximum(z, 0)
        err = x - xh
        loss = float(np.mean(np.sum(importance * err * err, axis=1)))
        dz = -2 * importance * err * (z > 0) / len(x)
        dw = h.T @ dz + (dz @ w.T).T @ x
        opt.step({"W": dw, "b": dz.sum(axis=0)})
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "loss": loss})
    w, b = params["W"], params["b"]
    norms = np.linalg.norm(w, axis=0)
    unit = w / np.where(norms == 0, 1, norms)
    # Elhage et al.'s "dimensions per feature": how much of a dimension each feature gets
    interference = (unit.T @ w) ** 2
    dims = norms ** 2 / np.maximum(interference.sum(axis=1), 1e-9)
    gram = w.T @ w
    return {
        "W": w.round(4).tolist(),
        "b": b.round(4).tolist(),
        "norms": norms.round(4).tolist(),
        "importance": importance.round(4).tolist(),
        "dims_per_feature": dims.round(4).tolist(),
        "features_represented": int((norms > 0.5).sum()),
        "gram": gram.round(4).tolist(),
        "history": history,
    }
