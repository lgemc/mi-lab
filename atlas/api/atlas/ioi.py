"""Indirect object identification in a transformer with six hand-written heads

"When Mary and John went to the store, John gave a drink to" -> Mary. Wang et
al. (2023) found GPT-2 small does this with a circuit of ~26 heads in a few
classes. This model *is* that circuit, one head per class, written by hand into
an attention-only transformer so that every weight is a sentence:

    L0.0  previous-token head     writes "the token before me" (IOI does not need it)
    L0.1  duplicate-token head    at the second John: "I have been seen before"
    L1.0  S-inhibition head       at the end: finds the duplicate, writes "not John"
    L1.1  name-agnostic copier    at the end: a little of every name, repeats count twice
    L2.0  name mover              at the end: attends to the name *not* inhibited, copies it
    L2.1  negative name mover     same attention, writes against it (GPT-2 has these too)

The residual stream is a set of named subspaces rather than a learned basis:
token identity, position, a few flags, "previous token", "inhibited name" and
"output name". The unembedding reads only the last, so the logit lens can only
see a name once a name mover has put it there.

Attention-only and no layer norm is a choice, not an omission: it makes the
direct logit attribution *exact* (the logits are a sum of head outputs), so the
page can show a decomposition that adds up rather than one that nearly does.

Real softmax attention runs on real matrices here. Patching, ablation, path
patching, edge attribution and the ACDC loop all intervene on this forward pass
and nothing is precomputed.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np

WORDS = ["<bos>", "When", "and", "went", "to", "the", "store", ",", "gave", "a", "drink"]
NAMES = ["Mary", "John", "Alice", "Bob", "Sara", "Tom"]
VOCAB = WORDS + NAMES
TOK = {w: i for i, w in enumerate(VOCAB)}
V, N = len(VOCAB), len(NAMES)
T = 15  # tokens in every template, <bos> included
BETA = 8.0  # sharpness of the hand-written attention
UNEMBED_SCALE = 4.0


class Layout:
    """Where each named subspace sits in the residual stream"""

    def __init__(self):
        cursor = 0
        self.slices: Dict[str, slice] = {}
        for name, width in [("tok", V), ("pos", T), ("const", 1), ("isname", 1), ("bos", 1), ("dup", 1),
                            ("prev", V), ("inhib", N), ("out", N)]:
            self.slices[name] = slice(cursor, cursor + width)
            cursor += width
        self.d = cursor

    def idx(self, name: str, offset: int = 0) -> int:
        return self.slices[name].start + offset


L = Layout()
D = L.d


@dataclass(frozen=True)
class Head:
    name: str
    layer: int
    role: str
    summary: str


HEADS = [
    Head("L0.0", 0, "previous token", "Writes the identity of the token one position back. Nothing in IOI reads it."),
    Head("L0.1", 0, "duplicate token", "At a name that appeared before, attends to its first occurrence and writes "
                                       "a 'duplicate' flag."),
    Head("L1.0", 1, "S-inhibition", "Attends to whatever carries the duplicate flag and writes that name into the "
                                    "'inhibit' subspace: the repeated name is the subject, so do not say it."),
    Head("L1.1", 1, "name-agnostic copier", "Spreads attention over every name and copies a little of each to the "
                                            "output. A repeated name gets copied twice, so this head leans slightly "
                                            "the wrong way."),
    Head("L2.0", 2, "name mover", "Attends to names, minus the inhibited one, and copies what it finds to the output. "
                                  "This head is where the answer is written."),
    Head("L2.1", 2, "negative name mover", "Same attention as the name mover, writes the opposite sign: a hedge "
                                           "against over-confidence, like the ones Wang et al. found in GPT-2."),
]
HEAD_NAMES = [h.name for h in HEADS]
LAYERS = 3


def _qk(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """The query and key maps of one head, as d x k matrices"""
    if name == "L0.0":
        wq, wk = np.zeros((D, T)), np.zeros((D, T))
        for i in range(1, T):
            wq[L.idx("pos", i), i - 1] = 1
        for j in range(T):
            wk[L.idx("pos", j), j] = 1
        return wq, wk
    if name == "L0.1":
        k = N + T + 1
        wq, wk = np.zeros((D, k)), np.zeros((D, k))
        for n in range(N):
            wq[L.idx("tok", len(WORDS) + n), n] = 1
            wk[L.idx("tok", len(WORDS) + n), n] = 1
        for i in range(T):
            wq[L.idx("pos", i), N + i] = -1  # a token does not count as its own duplicate
            wk[L.idx("pos", i), N + i] = 1
        wq[L.idx("const"), k - 1] = 0.5  # fall back to <bos> when nothing matches
        wk[L.idx("bos"), k - 1] = 1
        return wq, wk
    if name == "L1.0":
        wq, wk = np.zeros((D, 2)), np.zeros((D, 2))
        wq[L.idx("const"), 0], wk[L.idx("dup"), 0] = 1, 1
        wq[L.idx("const"), 1], wk[L.idx("bos"), 1] = 0.5, 1
        return wq, wk
    if name == "L1.1":
        wq, wk = np.zeros((D, 1)), np.zeros((D, 1))
        wq[L.idx("const"), 0], wk[L.idx("isname"), 0] = 3 / BETA, 1  # soft: a spread, not a pointer
        return wq, wk
    if name in ("L2.0", "L2.1"):
        wq, wk = np.zeros((D, 1 + N)), np.zeros((D, 1 + N))
        wq[L.idx("const"), 0], wk[L.idx("isname"), 0] = 1, 1
        for n in range(N):
            wq[L.idx("inhib", n), 1 + n] = -2
            wk[L.idx("tok", len(WORDS) + n), 1 + n] = 1
        return wq, wk
    raise KeyError(name)


def _ov(name: str) -> np.ndarray:
    """The output-value circuit of one head, W_V W_O, as one d x d matrix"""
    m = np.zeros((D, D))
    if name == "L0.0":
        for v in range(V):
            m[L.idx("tok", v), L.idx("prev", v)] = 1
    elif name == "L0.1":
        m[L.idx("isname"), L.idx("dup")] = 1
    elif name == "L1.0":
        for n in range(N):
            m[L.idx("tok", len(WORDS) + n), L.idx("inhib", n)] = 1
    elif name == "L1.1":
        for n in range(N):
            m[L.idx("tok", len(WORDS) + n), L.idx("out", n)] = 0.15
    elif name == "L2.0":
        for n in range(N):
            m[L.idx("tok", len(WORDS) + n), L.idx("out", n)] = 1
    elif name == "L2.1":
        for n in range(N):
            m[L.idx("tok", len(WORDS) + n), L.idx("out", n)] = -0.35
    return m


WEIGHTS = {h: (*_qk(h), _ov(h)) for h in HEAD_NAMES}
UNEMBED = np.zeros((D, N))
for _n in range(N):
    UNEMBED[L.idx("out", _n), _n] = UNEMBED_SCALE


# --------------------------------------------------------------------------- prompts


def tokens(io: str, s: str, template: str = "ABBA", s2: Optional[str] = None) -> List[str]:
    """The fixed frame, with the names put in; s2 replaces the second subject (the ABC corruption)"""
    first, second = (io, s) if template == "ABBA" else (s, io)
    return ["<bos>", "When", first, "and", second, "went", "to", "the", "store", ",", s2 or s,
            "gave", "a", "drink", "to"]


def embed(toks: List[str]) -> np.ndarray:
    x = np.zeros((len(toks), D))
    for i, t in enumerate(toks):
        x[i, L.idx("tok", TOK[t])] = 1
        x[i, L.idx("pos", i)] = 1
        x[i, L.idx("const")] = 1
        if t in NAMES:
            x[i, L.idx("isname")] = 1
        if t == "<bos>":
            x[i, L.idx("bos")] = 1
    return x


def _attend(xq: np.ndarray, xk: np.ndarray, xv: np.ndarray, head: str) -> Tuple[np.ndarray, np.ndarray]:
    wq, wk, ov = WEIGHTS[head]
    scores = BETA * (xq @ wq) @ (xk @ wk).T
    scores = np.where(np.tril(np.ones_like(scores)) > 0, scores, -np.inf)
    scores -= scores.max(axis=1, keepdims=True)
    pattern = np.exp(scores)
    pattern /= pattern.sum(axis=1, keepdims=True)
    return pattern, pattern @ (xv @ ov)


# An edge is (sender, receiver, channel): sender "embed" or a head, receiver a head or "logits",
# channel q/k/v for a head and "resid" for the logits.
EdgeKey = Tuple[str, str, str]


def run(toks: List[str], head_override: Optional[Dict[str, np.ndarray]] = None,
        edge_override: Optional[Dict[EdgeKey, np.ndarray]] = None) -> Dict:
    """One forward pass, with any head's output replaced and any edge's contribution replaced

    Each head reads its three inputs separately, each the sum of the embedding
    and every earlier head's output. That sum is written out edge by edge rather
    than as one residual tensor because path patching and edge attribution need
    to swap exactly one term of it.
    """
    head_override = head_override or {}
    edge_override = edge_override or {}
    x0 = embed(toks)
    outs: Dict[str, np.ndarray] = {}
    patterns: Dict[str, np.ndarray] = {}

    def incoming(receiver: str, channel: str, layer: int) -> np.ndarray:
        total = edge_override.get(("embed", receiver, channel), x0).copy()
        for h in HEADS:
            if h.layer < layer:
                total += edge_override.get((h.name, receiver, channel), outs[h.name])
        return total

    resid_after = [x0.copy()]
    for layer in range(LAYERS):
        for h in (h for h in HEADS if h.layer == layer):
            pattern, out = _attend(incoming(h.name, "q", layer), incoming(h.name, "k", layer),
                                   incoming(h.name, "v", layer), h.name)
            patterns[h.name] = pattern
            outs[h.name] = head_override.get(h.name, out)
        resid_after.append(x0 + sum(outs[h.name] for h in HEADS if h.layer <= layer))
    final = incoming("logits", "resid", LAYERS)
    logits = final[-1] @ UNEMBED
    return {"x0": x0, "outs": outs, "patterns": patterns, "final": final, "logits": logits,
            "resid_after": resid_after}


def logit_diff(result: Dict, io: str, s: str) -> float:
    lg = result["logits"]
    return float(lg[NAMES.index(io)] - lg[NAMES.index(s)])


@lru_cache(maxsize=4)
def dataset(template: str = "both") -> List[Tuple[str, str, str]]:
    templates = ["ABBA", "BABA"] if template == "both" else [template]
    return [(io, s, t) for io, s in permutations(NAMES, 2) for t in templates]


@lru_cache(maxsize=1)
def mean_outputs() -> Dict[str, np.ndarray]:
    """Each head's output averaged over every prompt, per position: what a mean ablation writes"""
    runs = [run(tokens(io, s, t))["outs"] for io, s, t in dataset()]
    return {h: np.mean([r[h] for r in runs], axis=0) for h in HEAD_NAMES}


def third_name(io: str, s: str) -> str:
    return next(n for n in NAMES if n not in (io, s))


def corrupted(io: str, s: str, template: str, kind: str) -> List[str]:
    """abc: the second subject becomes a fresh name, so nothing is duplicated.
    swap: the two roles exchange, so the right answer flips."""
    if kind == "abc":
        return tokens(io, s, template, s2=third_name(io, s))
    return tokens(s, io, template)


# ------------------------------------------------------------------------ the lessons' calls


def forward_view(io: str, s: str, template: str, ablate: Dict[str, str], corrupt: Optional[str] = None) -> Dict:
    """Attention patterns, direct logit attribution, the logit lens at the end, under ablations"""
    toks = corrupted(io, s, template, corrupt) if corrupt else tokens(io, s, template)
    override = {}
    means = mean_outputs()
    for h, mode in ablate.items():
        if mode == "zero":
            override[h] = np.zeros((T, D))
        elif mode == "mean":
            override[h] = means[h]
    r = run(toks, head_override=override)
    io_i, s_i = NAMES.index(io), NAMES.index(s)
    direction = UNEMBED[:, io_i] - UNEMBED[:, s_i]
    dla = {h: float(r["outs"][h][-1] @ direction) for h in HEAD_NAMES}
    lens = []
    for layer, resid in enumerate(r["resid_after"]):
        lg = resid[-1] @ UNEMBED
        p = np.exp(lg - lg.max())
        lens.append({"layer": layer, "probs": (p / p.sum()).round(4).tolist(),
                     "logit_diff": float(resid[-1] @ direction)})
    subspaces = {name: r["final"][-1][sl].round(3).tolist() for name, sl in L.slices.items()
                 if name in ("dup", "inhib", "out")}
    probs = np.exp(r["logits"] - r["logits"].max())
    return {
        "tokens": toks,
        "patterns": {h: r["patterns"][h].round(4).tolist() for h in HEAD_NAMES},
        "dla": dla,
        "logits": r["logits"].round(4).tolist(),
        "probs": (probs / probs.sum()).round(4).tolist(),
        "logit_diff": logit_diff(r, io, s),
        "lens": lens,
        "end_state": subspaces,
        "dup_flag": [float(v) for v in r["final"][:, L.idx("dup")].round(3)],
    }


def patch_heads(io: str, s: str, template: str, corrupt: str, direction: str) -> Dict:
    """Activation patching per head and per (head, position)

    denoise: corrupted run, one head's output restored from the clean run.
    noise:   clean run, one head's output taken from the corrupted run.
    Scores are fractions of the clean-to-corrupted logit-difference gap.
    """
    clean_t, corr_t = tokens(io, s, template), corrupted(io, s, template, corrupt)
    clean, corr = run(clean_t), run(corr_t)
    ld_clean, ld_corr = logit_diff(clean, io, s), logit_diff(corr, io, s)
    gap = ld_clean - ld_corr
    base_t, source = (corr_t, clean) if direction == "denoise" else (clean_t, corr)
    base = corr if direction == "denoise" else clean
    per_head, per_pos = {}, {}
    for h in HEAD_NAMES:
        ld = logit_diff(run(base_t, head_override={h: source["outs"][h]}), io, s)
        per_head[h] = _frac(ld, ld_clean, ld_corr, direction)
        row = []
        for pos in range(T):
            patched = base["outs"][h].copy()
            patched[pos] = source["outs"][h][pos]
            row.append(_frac(logit_diff(run(base_t, head_override={h: patched}), io, s), ld_clean, ld_corr, direction))
        per_pos[h] = row
    return {"clean_tokens": clean_t, "corrupt_tokens": corr_t, "ld_clean": ld_clean, "ld_corrupt": ld_corr,
            "gap": gap, "per_head": per_head, "per_position": per_pos}


def _frac(ld: float, ld_clean: float, ld_corr: float, direction: str) -> float:
    gap = ld_clean - ld_corr
    if abs(gap) < 1e-9:
        return 0.0
    return (ld - ld_corr) / gap if direction == "denoise" else (ld_clean - ld) / gap


def edge_list() -> List[EdgeKey]:
    edges: List[EdgeKey] = []
    senders = ["embed", *HEAD_NAMES]
    for h in HEADS:
        for s in senders:
            if s == "embed" or next(x for x in HEADS if x.name == s).layer < h.layer:
                for c in ("q", "k", "v"):
                    edges.append((s, h.name, c))
    edges.extend((s, "logits", "resid") for s in senders)
    return edges


def edges(io: str, s: str, template: str, corrupt: str, tau: float) -> Dict:
    """Every edge scored three ways, and the circuit ACDC keeps at threshold tau

    - patch: noise one edge -- the receiver reads the sender's *corrupted*
      output on that one input, everything else clean. The true effect.
    - eap: edge attribution patching (Syed et al., 2023), the first-order
      estimate of the same number: (corrupt - clean) sender output, dotted with
      the gradient of the logit difference at the receiver's input. One
      directional derivative per edge here; one backward pass for all of them
      in a real implementation. Blind wherever the softmax is saturated: a
      hard attention pattern has almost no gradient, so an edge that changes
      *where* a head looks scores near zero however much it matters.
    - eap_ig: the same product with the gradient averaged along the path from
      clean to corrupted (Hanna et al., 2024), which is what rescues those edges.
    - acdc: Conmy et al. (2023). Walk receivers from the logits backwards; for
      each incoming edge, patch it and keep it patched if the logit difference
      moves by less than tau. What survives is the circuit.
    """
    clean_t, corr_t = tokens(io, s, template), corrupted(io, s, template, corrupt)
    clean, corr = run(clean_t), run(corr_t)
    ld_clean = logit_diff(clean, io, s)
    ld_corr = logit_diff(corr, io, s)
    rows = []
    corrupt_out = {"embed": corr["x0"], **corr["outs"]}
    clean_out = {"embed": clean["x0"], **clean["outs"]}
    eps = 1e-3
    for e in edge_list():
        sender = e[0]
        patched = logit_diff(run(clean_t, edge_override={e: corrupt_out[sender]}), io, s)
        delta = corrupt_out[sender] - clean_out[sender]
        up = logit_diff(run(clean_t, edge_override={e: clean_out[sender] + eps * delta}), io, s)
        down = logit_diff(run(clean_t, edge_override={e: clean_out[sender] - eps * delta}), io, s)
        integrated = []
        for a in np.linspace(0.05, 0.95, 10):
            point = clean_out[sender] + a * delta
            hi = logit_diff(run(clean_t, edge_override={e: point + eps * delta}), io, s)
            lo = logit_diff(run(clean_t, edge_override={e: point - eps * delta}), io, s)
            integrated.append((hi - lo) / (2 * eps))
        rows.append({"sender": e[0], "receiver": e[1], "channel": e[2],
                     "patch": patched - ld_clean, "eap": (up - down) / (2 * eps),
                     "eap_ig": float(np.mean(integrated))})

    removed: Dict[EdgeKey, np.ndarray] = {}
    current = ld_clean
    receivers = ["logits"] + [h.name for h in sorted(HEADS, key=lambda h: -h.layer)]
    for r in receivers:
        for e in [e for e in edge_list() if e[1] == r]:
            trial = dict(removed)
            trial[e] = corrupt_out[e[0]]
            ld = logit_diff(run(clean_t, edge_override=trial), io, s)
            if abs(ld - current) < tau:
                removed = trial
                current = ld
    kept = [list(e) for e in edge_list() if e not in removed]
    return {"ld_clean": ld_clean, "ld_corrupt": ld_corr, "edges": rows, "acdc_kept": kept, "acdc_ld": current}


def meta() -> Dict:
    return {
        "names": NAMES,
        "heads": [h.__dict__ for h in HEADS],
        "layout": {k: [v.start, v.stop] for k, v in L.slices.items()},
        "d_model": D,
        "example": tokens("Mary", "John"),
    }
