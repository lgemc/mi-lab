"""A crosscoder between a "base" and a "chat" model, with the ground truth known

Model diffing asks what fine-tuning added. A crosscoder (Lindsey et al., 2024)
learns one dictionary for both models at once: each latent has one encoder
reading both activations and one decoder *per model*. A latent whose chat
decoder is large and base decoder near zero is a feature only chat uses.

The two "models" here are synthetic activations built from known features:

    shared   used by both, written in each model's own directions
    base     only the base model has (the fine-tune dropped it)
    chat     only the chat model has (the fine-tune added it)

so every latent can be graded. The page reads the relative decoder norm
r = |d_chat| / (|d_base| + |d_chat|): ~0 base-only, ~0.5 shared, ~1 chat-only.

The L1 penalty is where the method bites back (Minder et al., 2025): it
charges each latent for *both* decoder norms, so a shared feature that is a bit
stronger in chat can be cheaper to explain as a shared latent plus a spurious
"chat-only" one. BatchTopK (fixed number of active latents per batch, no L1)
removes that incentive, and the page lets you compare.
"""

from functools import lru_cache
from typing import Dict

import numpy as np

from .optim import Adam, rng, sparse_features

D_MODEL = 8


def ground_truth(n_shared: int, n_base: int, n_chat: int, drift: float):
    """Feature directions per model. Shared features differ between models by `drift`, and chat
    writes them a little louder, which is the situation that tempts the L1 penalty."""
    g = rng(21)
    n = n_shared + n_base + n_chat
    kinds = ["shared"] * n_shared + ["base"] * n_base + ["chat"] * n_chat
    d_base = g.normal(size=(D_MODEL, n))
    d_base /= np.linalg.norm(d_base, axis=0)
    wobble = g.normal(size=(D_MODEL, n))
    d_chat = d_base + drift * wobble
    d_chat /= np.linalg.norm(d_chat, axis=0)
    loud = np.array([1.3 if k == "shared" and i % 2 == 0 else 1.0 for i, k in enumerate(kinds)])
    d_chat = d_chat * loud
    base_mask = np.array([k != "chat" for k in kinds], float)
    chat_mask = np.array([k != "base" for k in kinds], float)
    return kinds, d_base * base_mask, d_chat * chat_mask


def sample(g, batch, kinds, d_base, d_chat, p=0.12):
    f = sparse_features(g, batch, len(kinds), p, 0.3, 1.0)
    return f, f @ d_base.T, f @ d_chat.T


@lru_cache(maxsize=64)
def train(mode: str = "l1", latents: int = 16, l1: float = 0.2, k: int = 2, drift: float = 0.2,
          n_shared: int = 6, n_base: int = 2, n_chat: int = 2, steps: int = 3000, seed: int = 0) -> Dict:
    kinds, db, dc = ground_truth(n_shared, n_base, n_chat, drift)
    g = rng(seed)
    p = {"W_b": g.normal(scale=0.3, size=(latents, D_MODEL)), "W_c": g.normal(scale=0.3, size=(latents, D_MODEL)),
         "b": np.zeros(latents),
         "D_b": g.normal(scale=0.3, size=(D_MODEL, latents)), "D_c": g.normal(scale=0.3, size=(D_MODEL, latents)),
         "c_b": np.zeros(D_MODEL), "c_c": np.zeros(D_MODEL)}
    opt = Adam(p, lr=5e-3)
    batch = 512
    for _ in range(steps):
        _, xb, xc = sample(g, batch, kinds, db, dc)
        a = xb @ p["W_b"].T + xc @ p["W_c"].T + p["b"]
        z = np.maximum(a, 0)
        if mode == "batchtopk":
            # keep the k*batch largest activations across the whole batch
            flat = z.ravel()
            cut = np.partition(flat, -k * batch)[-k * batch] if flat.size > k * batch else 0
            z = np.where(z >= max(cut, 1e-12), z, 0)
        eb = z @ p["D_b"].T + p["c_b"] - xb
        ec = z @ p["D_c"].T + p["c_c"] - xc
        nb = np.linalg.norm(p["D_b"], axis=0) + 1e-8
        nc = np.linalg.norm(p["D_c"], axis=0) + 1e-8
        dz = (2 * eb / batch) @ p["D_b"] + (2 * ec / batch) @ p["D_c"]
        g_db = (2 * eb / batch).T @ z
        g_dc = (2 * ec / batch).T @ z
        if mode == "l1":
            dz += l1 * (nb + nc) / batch
            g_db += l1 * z.mean(0) * p["D_b"] / nb
            g_dc += l1 * z.mean(0) * p["D_c"] / nc
        da = dz * (z > 0)
        opt.step({"W_b": da.T @ xb, "W_c": da.T @ xc, "b": da.sum(0), "D_b": g_db, "D_c": g_dc,
                  "c_b": (2 * eb / batch).sum(0), "c_c": (2 * ec / batch).sum(0)})
        if mode == "batchtopk":
            scale = np.sqrt(np.linalg.norm(p["D_b"], axis=0) ** 2 + np.linalg.norm(p["D_c"], axis=0) ** 2) + 1e-8
            p["D_b"] /= scale
            p["D_c"] /= scale

    _, xb, xc = sample(rng(seed + 500), 4000, kinds, db, dc)
    z = np.maximum(xb @ p["W_b"].T + xc @ p["W_c"].T + p["b"], 0)
    if mode == "batchtopk":
        cut = np.partition(z.ravel(), -k * len(z))[-k * len(z)]
        z = np.where(z >= max(cut, 1e-12), z, 0)
    alive = z.max(0) > 1e-6
    nb, nc = np.linalg.norm(p["D_b"], axis=0), np.linalg.norm(p["D_c"], axis=0)
    rel = nc / (nb + nc + 1e-12)
    # grade each latent against the truth: the feature whose (base, chat) decoder pair it matches best
    truth = np.concatenate([db, dc])
    truth_u = truth / (np.linalg.norm(truth, axis=0) + 1e-12)
    learned = np.concatenate([p["D_b"], p["D_c"]])
    learned_u = learned / (np.linalg.norm(learned, axis=0) + 1e-12)
    cos = learned_u.T @ truth_u  # [latents, features]
    match = cos.argmax(1)
    true_rel = np.linalg.norm(dc, axis=0) / (np.linalg.norm(db, axis=0) + np.linalg.norm(dc, axis=0))
    fvu = float((np.sum((z @ p["D_b"].T + p["c_b"] - xb) ** 2) + np.sum((z @ p["D_c"].T + p["c_c"] - xc) ** 2))
                / (np.sum((xb - xb.mean(0)) ** 2) + np.sum((xc - xc.mean(0)) ** 2)))
    lat = []
    for i in range(latents):
        lat.append({"id": i, "alive": bool(alive[i]), "rel": round(float(rel[i]), 4),
                    "norm_base": round(float(nb[i]), 4), "norm_chat": round(float(nc[i]), 4),
                    "match": int(match[i]), "match_kind": kinds[match[i]],
                    "match_cos": round(float(cos[i, match[i]]), 4),
                    "frequency": round(float((z[:, i] > 0).mean()), 4)})
    return {"latents": lat, "kinds": kinds, "true_rel": true_rel.round(4).tolist(), "fvu": fvu,
            "l0": float((z > 0).sum(1).mean()),
            "spurious_chat_only": sum(1 for x in lat if x["alive"] and x["rel"] > 0.9 and x["match_kind"] == "shared")}
