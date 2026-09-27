"""A two-hop residual network for comparing the logit lens, the Jacobian lens and a tuned lens

"The capital of the country where Curie was born is" -> Warsaw, by way of
Poland, which the model never writes. That silent middle step is what
WorkspaceBench asks a readout to find, and this model has one by construction.

The residual stream at the last position has four named parts:

    T  the input token (attention has already moved the subject here; that part is not modelled)
    W  a *workspace*: whatever the model is thinking about, stored in a rotated basis R e_x
    O  the output basis: the only part the unembedding reads besides a faint echo of T
    p  one flag, "a hop is still pending"

and six residual MLP blocks, each a handful of ReLUs:

    1 hop 1       subject in T and p on  ->  write R e_country into W
    2 noise       small random writes into W (models are not clean)
    3 hop 2       country in W and p on  ->  replace it with R e_capital, switch p off
    4 noise
    5 verbalize   p off  ->  copy W into O through R^T: say what you are thinking
    6 noise

The logit lens decodes each layer's residual as if it were the last. W is not
in the unembedding's view, so the logit lens sees only the echo of the subject
until block 5 speaks. The J-lens (Anthropic, 2026) first carries the residual to
the last layer with J_l = E[dh_final/dh_l], the Jacobian averaged over a corpus.
In most of the corpus p is off and block 5 is live, so J_l contains the
verbalizer R^T, and the J-lens reads *Poland* at layers 1 and 2: what the model
would say if it said what it is holding. A tuned lens (here, least squares from
h_l to h_final over the same corpus) learns much the same map from data.

Everything here is computed: the Jacobians by central differences through the
real blocks, averaged over a real sampled corpus, at import time.
"""

from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

SUBJECTS = ["Curie", "Turing", "Hokusai", "Kafka", "Gandhi"]
COUNTRIES = ["Poland", "England", "Japan", "Czechia", "India"]
CAPITALS = ["Warsaw", "London", "Tokyo", "Prague", "Delhi"]
FILLER = ["is", "the"]
VOCAB = SUBJECTS + COUNTRIES + CAPITALS + FILLER
V = len(VOCAB)
IDX = {w: i for i, w in enumerate(VOCAB)}

T_, W_, O_ = slice(0, V), slice(V, 2 * V), slice(2 * V, 3 * V)
P_ = 3 * V
D = 3 * V + 1
ECHO, SPEAK = 1.5, 6.0
N_BLOCKS = 6

_gen = np.random.default_rng(7)
R, _ = np.linalg.qr(_gen.normal(size=(V, V)))  # the workspace's private basis
NOISE = [(_gen.normal(scale=0.35, size=(8, V)), _gen.normal(scale=0.05, size=(V, 8))) for _ in range(3)]

BLOCKS = [
    ("hop 1", "subject + pending → country into the workspace"),
    ("noise", "small random writes into the workspace"),
    ("hop 2", "country + pending → capital replaces it; pending switches off"),
    ("noise", "small random writes into the workspace"),
    ("verbalize", "pending off → copy the workspace into the output basis"),
    ("noise", "small random writes into the workspace"),
]


def relu(x):
    return np.maximum(x, 0)


def block(i: int, h: np.ndarray) -> np.ndarray:
    """Apply block i (0-based) to a batch of residuals [n, D], returning the new residuals"""
    h = np.atleast_2d(h)
    out = h.copy()
    w_read = h[:, W_] @ R  # the workspace in token coordinates: R^T w, row-wise
    pending = h[:, P_]
    name = BLOCKS[i][0]
    if name == "hop 1":
        for s, c in zip(SUBJECTS, COUNTRIES, strict=True):
            u = relu(h[:, IDX[s]] + pending - 1)
            out[:, W_] += np.outer(u, R[:, IDX[c]])
    elif name == "hop 2":
        for c, cap in zip(COUNTRIES, CAPITALS, strict=True):
            u = relu(w_read[:, IDX[c]] + pending - 1)
            out[:, W_] += np.outer(u, R[:, IDX[cap]] - R[:, IDX[c]])
            out[:, P_] -= u
    elif name == "verbalize":
        u = relu(w_read - 2 * pending[:, None] + 0.1)
        out[:, O_] += u
    else:
        w_in, w_out = NOISE[i // 2]
        out[:, W_] += relu(h[:, W_] @ w_in.T) @ w_out.T
    return out


def forward(h0: np.ndarray, patch: Optional[Dict] = None) -> List[np.ndarray]:
    """Residuals after each block, h_0 .. h_6, optionally overwriting the workspace after one layer"""
    hs = [np.atleast_2d(h0).copy()]
    for i in range(N_BLOCKS):
        h = block(i, hs[-1])
        if patch and patch["layer"] == i + 1:
            h = h.copy()
            h[:, W_] = patch["workspace"]
        hs.append(h)
    return hs


def unembed(h: np.ndarray) -> np.ndarray:
    """Final RMSNorm, then the unembedding: the output basis loudly, the input echo faintly"""
    h = np.atleast_2d(h)
    n = h / np.sqrt(np.mean(h * h, axis=1, keepdims=True) + 1e-6)
    return SPEAK * n[:, O_] + ECHO * n[:, T_]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def state(token: str, pending: bool, thinking: Optional[str] = None) -> np.ndarray:
    h = np.zeros(D)
    h[IDX[token]] = 1
    h[P_] = float(pending)
    if thinking:
        h[W_] = R[:, IDX[thinking]]
    return h


CORPUS_MIXES = (0.0, 0.03, 0.08, 0.25)


@lru_cache(maxsize=len(CORPUS_MIXES))
def corpus(two_hop: float = 0.0, n: int = 400) -> np.ndarray:
    """The inputs the lenses are fitted on: a toy stand-in for web text

    Most of the time the model says what it is holding (pending off, something
    in the workspace); a share `two_hop` of the time it faces a two-hop
    question; otherwise it has nothing in mind at all. The lens only knows the
    model through these, and that share turns out to be the whole story of
    whether a lens reads the middle step or skips ahead to the answer.
    """
    g = np.random.default_rng(11)
    rows = []
    for _ in range(n):
        r = g.random()
        tok = VOCAB[g.integers(V)]
        if r < two_hop:
            rows.append(state(SUBJECTS[g.integers(len(SUBJECTS))], True))
        elif r < two_hop + 0.6:
            rows.append(state(tok, False, VOCAB[g.integers(V)]))
        else:
            rows.append(state(tok, False))
    return np.array(rows)


def _jacobian_to_final(layer: int, h: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """d h_6 / d h_layer at one point, by central differences through the remaining blocks"""
    probes = np.concatenate([h + eps * np.eye(D), h - eps * np.eye(D)])
    for i in range(layer, N_BLOCKS):
        probes = block(i, probes)
    return (probes[:D] - probes[D:]).T / (2 * eps)  # [D_out, D_in]


@lru_cache(maxsize=len(CORPUS_MIXES))
def fitted(two_hop: float = 0.0) -> Dict[str, List[np.ndarray]]:
    """The J-lens (mean Jacobian per layer) and a tuned lens (least-squares affine map per layer)"""
    hs = forward(corpus(two_hop))
    final = hs[-1]
    jac, tuned = [], []
    for layer in range(N_BLOCKS + 1):
        if layer == N_BLOCKS:
            jac.append(np.eye(D))
            tuned.append(np.vstack([np.eye(D), np.zeros((1, D))]))
            continue
        jac.append(np.mean([_jacobian_to_final(layer, h) for h in hs[layer]], axis=0))
        x = np.hstack([hs[layer], np.ones((len(final), 1))])
        a = np.linalg.solve(x.T @ x + 1e-3 * np.eye(D + 1), x.T @ final)
        tuned.append(a)
    return {"jlens": jac, "tuned": tuned}


def _read(p: np.ndarray, k: int = 5) -> List[Dict]:
    order = np.argsort(-p)[:k]
    return [{"token": VOCAB[i], "p": round(float(p[i]), 4)} for i in order]


def probe(subject: str, patch_country: Optional[str] = None, patch_layer: int = 2, two_hop: float = 0.0) -> Dict:
    """Read every layer three ways for one two-hop question, optionally editing the workspace

    An edit writes a different country into W after `patch_layer`. If the final
    answer follows the edit, the thing the J-lens read was not a coincidence of
    the readout -- the model uses it.
    """
    i = SUBJECTS.index(subject)
    patch = None
    if patch_country:
        patch = {"layer": patch_layer, "workspace": R[:, IDX[patch_country]]}
    hs = forward(state(subject, True), patch)
    lenses = fitted(two_hop)
    tracked = [subject, COUNTRIES[i], CAPITALS[i]] + ([patch_country, CAPITALS[COUNTRIES.index(patch_country)]]
                                                       if patch_country else [])
    layers = []
    for layer, h in enumerate(hs):
        views = {
            "logit": softmax(unembed(h))[0],
            "jlens": softmax(unembed(h @ lenses["jlens"][layer].T))[0],
            "tuned": softmax(unembed(np.hstack([h, np.ones((1, 1))]) @ lenses["tuned"][layer]))[0],
        }
        layers.append({
            "layer": layer,
            "after": "embedding" if layer == 0 else BLOCKS[layer - 1][0],
            "top": {k: _read(v) for k, v in views.items()},
            "tracked": {k: {t: round(float(v[IDX[t]]), 4) for t in dict.fromkeys(tracked)} for k, v in views.items()},
            "workspace": _read(np.abs(h[0, W_] @ R), 3),
            "pending": round(float(h[0, P_]), 3),
        })
    final = softmax(unembed(hs[-1]))[0]
    return {"subject": subject, "country": COUNTRIES[i], "capital": CAPITALS[i], "layers": layers,
            "answer": _read(final, 3), "tracked": list(dict.fromkeys(tracked))}


def meta() -> Dict:
    return {"subjects": SUBJECTS, "countries": COUNTRIES, "capitals": CAPITALS, "vocab": VOCAB,
            "blocks": [{"name": n, "does": d} for n, d in BLOCKS], "d_model": D,
            "corpus_size": len(corpus()), "corpus_mixes": list(CORPUS_MIXES)}
