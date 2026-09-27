"""A sparse autoencoder trained on 2-D data made of known sparse features

The data is the superposition picture turned around: k feature directions in
the plane, each on with probability p at a random magnitude, summed. A point
is a sum of a few arrows, so the cloud is a star. The SAE sees only the points
and has to find the arrows:

    z  = act(W_enc (x - b_dec) + b_enc)
    x' = W_dec z + b_dec

with `act` either ReLU plus an L1 penalty weighted by decoder norms (Anthropic's
April 2024 recipe) or TopK (Gao et al., 2024), which fixes L0 by construction
and needs no penalty. Because the truth is known, the page can score what
real SAEs never can be scored on: which true feature each latent found, which
it missed, and which latents are dead or split one feature in two.
"""

from functools import lru_cache
from typing import Dict

import numpy as np

from .optim import Adam, rng, sparse_features


def true_directions(k: int) -> np.ndarray:
    """k unit vectors spread around the circle, slightly irregular so no two are symmetric by accident"""
    angles = np.linspace(0, np.pi * 2, k, endpoint=False) + 0.25 + 0.15 * np.sin(np.arange(k) * 1.7)
    return np.stack([np.cos(angles), np.sin(angles)])  # [2, k]


def sample(g: np.random.Generator, batch: int, k: int, p: float, noise: float) -> (np.ndarray, np.ndarray):
    f = sparse_features(g, batch, k, p, 0.3, 1.0)
    x = f @ true_directions(k).T + g.normal(scale=noise, size=(batch, 2))
    return x, f


@lru_cache(maxsize=256)
def train(k_true: int = 5, latents: int = 5, p: float = 0.15, l1: float = 0.3, mode: str = "relu",
          topk: int = 1, noise: float = 0.01, steps: int = 3000, seed: int = 0) -> Dict:
    g = rng(seed)
    dirs = true_directions(k_true)
    params = {
        "W_enc": g.normal(scale=0.5, size=(latents, 2)),
        "b_enc": np.zeros(latents),
        "W_dec": np.zeros((2, latents)),
        "b_dec": np.zeros(2),
    }
    params["W_dec"] = params["W_enc"].T / np.linalg.norm(params["W_enc"], axis=1)  # tied at init, a common trick
    # b_dec starts at the data's median, as the reference implementations do. Under TopK it also stays
    # there: with one latent per input and a free centre, the centre walks off the cloud and the latents
    # fan out from it like a spotlight -- a fine reconstruction that has found no feature at all.
    params["b_dec"] = np.median(sample(rng(seed + 7), 2048, k_true, p, noise)[0], axis=0)
    opt = Adam(params, lr=5e-3)
    history = []
    fired = np.zeros(latents)
    resampled = 0
    for step in range(steps):
        x, _ = sample(g, 512, k_true, p, noise)
        we, be, wd, bd = params["W_enc"], params["b_enc"], params["W_dec"], params["b_dec"]
        c = x - bd
        a = c @ we.T + be
        z = np.maximum(a, 0)
        if mode == "topk":
            keep = np.argsort(-z, axis=1)[:, :topk]
            mask = np.zeros_like(z)
            np.put_along_axis(mask, keep, 1.0, axis=1)
            z = z * mask
        xh = z @ wd.T + bd
        err = xh - x
        norms = np.linalg.norm(wd, axis=0) + 1e-8
        mse = float(np.mean(np.sum(err * err, axis=1)))
        penalty = l1 * float(np.mean(z @ norms)) if mode == "relu" else 0.0
        dxh = 2 * err / len(x)
        dwd = dxh.T @ z
        dz = dxh @ wd
        if mode == "relu":
            dwd += l1 * z.mean(axis=0) * wd / norms
            dz += l1 * norms / len(x)
        da = dz * (z > 0)
        dbd = dxh.sum(axis=0) - (da @ we).sum(axis=0)
        grads = {"W_enc": da.T @ c, "b_enc": da.sum(axis=0), "W_dec": dwd}
        if mode != "topk":
            grads["b_dec"] = dbd
        opt.step(grads)
        if mode == "topk":
            params["W_dec"] /= np.linalg.norm(params["W_dec"], axis=0, keepdims=True) + 1e-8
        fired += (z > 0).sum(axis=0)
        if step % 250 == 249 and step < steps * 0.7:
            # Resample the dead (Bricken et al., 2023): a latent that never fired is pointed at a
            # badly reconstructed input, so it gets a second chance instead of wasting a slot.
            worst = np.argsort(-np.sum(err * err, axis=1))
            for j, lat in enumerate(np.flatnonzero(fired == 0)):
                target = x[worst[j % len(worst)]] - bd
                target = target / (np.linalg.norm(target) + 1e-8)
                params["W_dec"][:, lat] = target
                params["W_enc"][lat] = 0.5 * target
                params["b_enc"][lat] = 0.0
                resampled += 1
            fired[:] = 0
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "mse": mse, "penalty": penalty, "l0": float((z > 0).sum(axis=1).mean())})

    eval_x, eval_f = sample(rng(seed + 1000), 4000, k_true, p, noise)
    we, be, wd, bd = params["W_enc"], params["b_enc"], params["W_dec"], params["b_dec"]
    z = np.maximum((eval_x - bd) @ we.T + be, 0)
    if mode == "topk":
        keep = np.argsort(-z, axis=1)[:, :topk]
        mask = np.zeros_like(z)
        np.put_along_axis(mask, keep, 1.0, axis=1)
        z = z * mask
    xh = z @ wd.T + bd
    fvu = float(np.sum((xh - eval_x) ** 2) / np.sum((eval_x - eval_x.mean(axis=0)) ** 2))
    alive = z.max(axis=0) > 1e-6
    wd_unit = wd / (np.linalg.norm(wd, axis=0, keepdims=True) + 1e-8)
    cos = dirs.T @ wd_unit  # [k_true, latents]
    cos = np.where(alive[None, :], cos, -1)
    shown = 400
    return {
        "true": dirs.T.round(4).tolist(),
        "decoder": wd.T.round(4).tolist(),
        "encoder": we.round(4).tolist(),
        "b_enc": be.round(4).tolist(),
        "b_dec": bd.round(4).tolist(),
        "alive": alive.tolist(),
        "cos": cos.round(4).tolist(),
        "best_match": cos.max(axis=1).round(4).tolist(),
        "l0": float((z > 0).sum(axis=1).mean()),
        "fvu": fvu,
        "dead": int((~alive).sum()),
        "resampled": resampled,
        "points": eval_x[:shown].round(4).tolist(),
        "point_latents": z[:shown].round(3).tolist(),
        "point_features": eval_f[:shown].round(3).tolist(),
        "history": history,
    }


def sweep(k_true: int, latents: int, p: float, mode: str) -> Dict:
    """The sparsity/fidelity frontier: one SAE per penalty (or k), each point is (L0, unexplained variance)"""
    rows = []
    if mode == "topk":
        for k in range(1, min(latents, 4) + 1):
            r = train(k_true, latents, p, 0.0, "topk", k, steps=1500)
            rows.append({"knob": k, "l0": r["l0"], "fvu": r["fvu"], "dead": r["dead"]})
    else:
        for l1 in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            r = train(k_true, latents, p, l1, "relu", 1, steps=1500)
            rows.append({"knob": l1, "l0": r["l0"], "fvu": r["fvu"], "dead": r["dead"]})
    return {"rows": rows}
