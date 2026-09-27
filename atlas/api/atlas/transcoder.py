"""A transcoder replacing a small MLP whose neurons are polysemantic

The MLP is trained, not wired: five sparse input features, four hidden
neurons, five output features to compute --

    out_a = a            out_b = b          out_and = min(a, b)
    out_c = c            out_d_not_e = relu(d - e)

-- which does not fit in four neurons one-per-job, so the neurons it learns are
mixtures. A neuron is then a bad unit of explanation: its input weights read
three features and its output writes to four.

A transcoder (Dunefsky et al., 2024) is trained to *imitate* the MLP, reading
its input and predicting its output through many sparse latents:

    z = ReLU(W_enc x + b_enc),   y' = W_dec z + b_dec,   loss = |y - y'|^2 + l1 * sum_i |z_i| |W_dec,i|

Unlike an SAE on the hidden layer, it can stand in for the MLP, so its latents
form a graph with the features upstream and downstream whose edge weights do
not depend on the input: latent i reads input feature j with weight
W_enc,i . e_j, and writes output feature k with weight u_k . W_dec,i. For one
input, multiply by the activations and the result is an attribution graph --
the construction Anthropic's circuit tracing (Ameisen et al., 2025) scales up
with cross-layer transcoders.
"""

from functools import lru_cache
from typing import Dict, List

import numpy as np

from .optim import Adam, rng, sparse_features

IN_FEATURES = ["a", "b", "c", "d", "e"]
OUT_FEATURES = ["a", "b", "a AND b", "c", "d NOT e"]
D_IN, D_OUT, HIDDEN = 5, 5, 4
P_ACTIVE = np.array([0.35, 0.35, 0.2, 0.25, 0.25])


def target(f: np.ndarray) -> np.ndarray:
    a, b, c, d, e = f.T
    return np.stack([a, b, np.minimum(a, b), c, np.maximum(d - e, 0)], axis=1)


@lru_cache(maxsize=1)
def geometry():
    """Input and output feature directions: random orthonormal bases, so no neuron is aligned by accident"""
    g = rng(3)
    e_in, _ = np.linalg.qr(g.normal(size=(D_IN, D_IN)))
    e_out, _ = np.linalg.qr(g.normal(size=(D_OUT, D_OUT)))
    return e_in, e_out  # columns are feature directions


def data(g: np.random.Generator, batch: int):
    e_in, e_out = geometry()
    f = sparse_features(g, batch, len(IN_FEATURES), P_ACTIVE, 0.4, 1.0)
    return f, f @ e_in.T, target(f) @ e_out.T


@lru_cache(maxsize=1)
def mlp() -> Dict[str, np.ndarray]:
    """Train the MLP being explained: x -> W2 ReLU(W1 x + b1) + b2"""
    g = rng(0)
    # A small positive bias keeps every neuron alive at the start; with a zero one, seeds lose a
    # neuron in the first steps and the MLP quietly stops computing `c` at all.
    p = {"W1": g.normal(scale=0.5, size=(HIDDEN, D_IN)), "b1": np.full(HIDDEN, 0.1),
         "W2": g.normal(scale=0.5, size=(D_OUT, HIDDEN)), "b2": np.zeros(D_OUT)}
    opt = Adam(p, lr=1e-2)
    for step in range(6000):
        if step == 4500:
            opt.lr = 2e-3
        _, x, y = data(g, 512)
        a = x @ p["W1"].T + p["b1"]
        h = np.maximum(a, 0)
        yh = h @ p["W2"].T + p["b2"]
        dy = 2 * (yh - y) / len(x)
        da = (dy @ p["W2"]) * (a > 0)
        opt.step({"W1": da.T @ x, "b1": da.sum(0), "W2": dy.T @ h, "b2": dy.sum(0)})
    return p


def run_mlp(x: np.ndarray):
    p = mlp()
    h = np.maximum(x @ p["W1"].T + p["b1"], 0)
    return h, h @ p["W2"].T + p["b2"]


@lru_cache(maxsize=64)
def train(latents: int = 8, l1: float = 0.1, steps: int = 3000, seed: int = 0) -> Dict:
    g = rng(seed)
    p = {"W_enc": g.normal(scale=0.4, size=(latents, D_IN)), "b_enc": np.zeros(latents),
         "W_dec": g.normal(scale=0.4, size=(D_OUT, latents)), "b_dec": np.zeros(D_OUT)}
    opt = Adam(p, lr=5e-3)
    for _ in range(steps):
        _, x, _ = data(g, 512)
        _, y = run_mlp(x)  # the transcoder imitates the MLP, not the task
        a = x @ p["W_enc"].T + p["b_enc"]
        z = np.maximum(a, 0)
        yh = z @ p["W_dec"].T + p["b_dec"]
        norms = np.linalg.norm(p["W_dec"], axis=0) + 1e-8
        dy = 2 * (yh - y) / len(x)
        dz = dy @ p["W_dec"] + l1 * norms / len(x)
        dwd = dy.T @ z + l1 * z.mean(0) * p["W_dec"] / norms
        da = dz * (a > 0)
        opt.step({"W_enc": da.T @ x, "b_enc": da.sum(0), "W_dec": dwd, "b_dec": dy.sum(0)})
    return _describe(p)


def _describe(p: Dict[str, np.ndarray]) -> Dict:
    e_in, e_out = geometry()
    mw = mlp()
    _, x, y_task = data(rng(99), 4000)
    h, y = run_mlp(x)
    z = np.maximum(x @ p["W_enc"].T + p["b_enc"], 0)
    yh = z @ p["W_dec"].T + p["b_dec"]
    fvu = float(np.sum((yh - y) ** 2) / np.sum((y - y.mean(0)) ** 2))
    mlp_err = float(np.sum((y - y_task) ** 2) / np.sum((y_task - y_task.mean(0)) ** 2))
    alive = z.max(0) > 1e-6
    return {
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        # what each unit reads from each input feature, and writes to each output feature
        "neuron_in": (mw["W1"] @ e_in).round(3).tolist(),
        "neuron_out": (e_out.T @ mw["W2"]).T.round(3).tolist(),
        "latent_in": (p["W_enc"] @ e_in).round(3).tolist(),
        "latent_out": (e_out.T @ p["W_dec"]).T.round(3).tolist(),
        "latent_bias": p["b_enc"].round(3).tolist(),
        "alive": alive.tolist(),
        "fvu": fvu,
        "mlp_task_fvu": mlp_err,
        "l0": float((z > 0).sum(1).mean()),
        "neuron_l0": float((h > 0).sum(1).mean()),
        "_params": {k: v.tolist() for k, v in p.items()},
    }


def attribution_graph(features: List[float], latents: int = 8, l1: float = 0.1) -> Dict:
    """For one input: input features -> active transcoder latents -> output features, plus the error

    Edge weight = activation of the source times the (input-independent) virtual weight.
    The error node is what the transcoder fails to reproduce of the MLP's real output.
    """
    e_in, e_out = geometry()
    tc = train(latents, l1)
    p = {k: np.array(v) for k, v in tc["_params"].items()}
    f = np.array(features, dtype=float)
    x = e_in @ f
    _, y = run_mlp(x[None])
    z = np.maximum(p["W_enc"] @ x + p["b_enc"], 0)
    yh = p["W_dec"] @ z + p["b_dec"]
    in_edges, out_edges = [], []
    for i in np.flatnonzero(z > 1e-6):
        for j, name in enumerate(IN_FEATURES):
            w = float(f[j] * (p["W_enc"][i] @ e_in[:, j]))
            if abs(w) > 1e-3:
                in_edges.append({"source": name, "latent": int(i), "weight": round(w, 4)})
        for k, name in enumerate(OUT_FEATURES):
            w = float(z[i] * (e_out[:, k] @ p["W_dec"][:, i]))
            if abs(w) > 1e-3:
                out_edges.append({"latent": int(i), "target": name, "weight": round(w, 4)})
    return {
        "latents": [{"id": int(i), "activation": round(float(z[i]), 4)} for i in np.flatnonzero(z > 1e-6)],
        "in_edges": in_edges,
        "out_edges": out_edges,
        "mlp_out": (e_out.T @ y[0]).round(4).tolist(),
        "transcoder_out": (e_out.T @ yh).round(4).tolist(),
        "bias_out": (e_out.T @ p["b_dec"]).round(4).tolist(),
        "error": (e_out.T @ (y[0] - yh)).round(4).tolist(),
        "task_out": target(f[None])[0].round(4).tolist(),
    }


def public(tc: Dict) -> Dict:
    return {k: v for k, v in tc.items() if not k.startswith("_")}
