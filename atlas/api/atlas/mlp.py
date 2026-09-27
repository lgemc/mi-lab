"""A seven-neuron network computing (a AND b) OR c, as a structural causal model

Every neuron is a node and every weight an arrow, so the causal vocabulary of
the first lessons -- intervene, mediate, patch -- applies to it word for word.
The weights are set by hand so that the network contains the two things that
make interpretability hard in real models, in the smallest form they come in:

- **Redundancy.** n1 and n2 both compute a AND b, and m1 ORs them. Knock out
  either one on a clean input and the output barely moves (the other covers
  for it); restore either one into a corrupted input and the output mostly
  comes back. Noising says "unimportant", denoising says "important", and both
  are right about different questions -- necessity versus sufficiency.
- **Saturation.** The output is a sigmoid. Once the logit is large, a gradient
  says nothing is important at all, which is exactly where attribution
  patching (a first-order estimate of patching) is wrong and integrated
  gradients earns its cost.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

INPUTS = ["a", "b", "c"]
LAYER1 = ["n1", "n2", "n3", "n4"]
LAYER2 = ["m1", "m2"]
NODES = INPUTS + LAYER1 + LAYER2 + ["out"]

# rows: neurons, columns: the layer below
W1 = np.array([
    [1.0, 1.0, 0.0],   # n1 = relu(a + b - 1): AND
    [1.0, 1.0, 0.0],   # n2 = the same AND, a backup copy
    [0.0, 0.0, 1.0],   # n3 = relu(c)
    [-1.0, 1.0, 0.0],  # n4 = relu(b - a): b without a, a distractor
])
B1 = np.array([-1.0, -1.0, 0.0, 0.0])
W2 = np.array([
    [1.0, 1.0, 1.0, 0.0],  # m1 = relu(n1 + n2 + n3): OR of the evidence
    [0.0, 0.0, 0.0, 1.0],  # m2 = relu(n4 - 0.2)
])
B2 = np.array([0.0, -0.2])
W3 = np.array([6.0, -1.0])
B3 = -3.0

DESCRIPTIONS = {
    "n1": "a AND b", "n2": "a AND b (backup copy)", "n3": "copies c", "n4": "b but not a (distractor)",
    "m1": "OR of n1, n2, n3", "m2": "passes n4 on, weakly", "out": "logit of the answer",
}


def edges() -> List[Dict]:
    out = []
    for i, n in enumerate(LAYER1):
        for j, s in enumerate(INPUTS):
            if W1[i, j]:
                out.append({"source": s, "target": n, "weight": float(W1[i, j])})
    for i, n in enumerate(LAYER2):
        for j, s in enumerate(LAYER1):
            if W2[i, j]:
                out.append({"source": s, "target": n, "weight": float(W2[i, j])})
    for j, s in enumerate(LAYER2):
        out.append({"source": s, "target": "out", "weight": float(W3[j])})
    return out


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def forward(x: Sequence[float], set_: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Every node's value, with do(node = v) applied for each entry of set_

    An intervention replaces the node's mechanism: its value is v whatever its
    parents say, and everything downstream is recomputed from it.
    """
    set_ = set_ or {}
    v = {n: float(set_.get(n, x[i])) for i, n in enumerate(INPUTS)}
    xin = np.array([v[n] for n in INPUTS])
    h1 = np.maximum(W1 @ xin + B1, 0)
    for i, n in enumerate(LAYER1):
        v[n] = float(set_.get(n, h1[i]))
    h1 = np.array([v[n] for n in LAYER1])
    h2 = np.maximum(W2 @ h1 + B2, 0)
    for i, n in enumerate(LAYER2):
        v[n] = float(set_.get(n, h2[i]))
    h2 = np.array([v[n] for n in LAYER2])
    v["out"] = float(set_.get("out", W3 @ h2 + B3))
    v["prob"] = sigmoid(v["out"])
    return v


def metric(values: Dict[str, float], kind: str) -> float:
    return values["prob"] if kind == "prob" else values["out"]


def patching(clean: Sequence[float], corrupt: Sequence[float], kind: str = "logit") -> Dict:
    """One-node patching both ways, attribution patching, and integrated gradients, per hidden node

    - denoise: run the corrupted input, restore one node to its clean value.
      Recovery 1 means the node alone is *sufficient* to bring the answer back.
    - noise: run the clean input, set one node to its corrupted value.
      Damage 1 means the node is *necessary*: without it the answer is gone.
    - attribution patching: the first-order Taylor estimate of denoising,
      (clean - corrupt) x d(metric)/d(node) taken at the corrupted run. One
      backward pass for every node at once, which is why people use it, and
      wrong exactly where the metric is curved.
    - integrated gradients: the same product with the gradient averaged along
      the straight line from corrupted to clean, which sums to the true total.
    """
    c = forward(clean)
    k = forward(corrupt)
    m_clean, m_corrupt = metric(c, kind), metric(k, kind)
    span = m_clean - m_corrupt
    hidden = LAYER1 + LAYER2
    rows = []
    for n in hidden:
        den = metric(forward(corrupt, {n: c[n]}), kind)
        noi = metric(forward(clean, {n: k[n]}), kind)
        delta = c[n] - k[n]
        grad = _grad(corrupt, n, k[n], kind)
        ig = delta * float(np.mean([_grad(corrupt, n, k[n] + a * delta, kind) for a in np.linspace(0.05, 0.95, 10)]))
        rows.append({
            "node": n,
            "clean": c[n], "corrupt": k[n],
            "denoise": den, "noise": noi,
            "recovery": (den - m_corrupt) / span if span else 0.0,
            "damage": (m_clean - noi) / span if span else 0.0,
            "attribution": delta * grad,
            "attribution_frac": delta * grad / span if span else 0.0,
            "integrated": ig,
            "true_effect": den - m_corrupt,
        })
    return {"clean": c, "corrupt": k, "metric_clean": m_clean, "metric_corrupt": m_corrupt, "rows": rows}


def _grad(x: Sequence[float], node: str, at: float, kind: str, eps: float = 1e-6) -> float:
    """d(metric)/d(node) with the node held at `at`, the way autograd would report it

    A one-sided (left) difference rather than a symmetric one, because of the
    kinks: at a ReLU's corner autograd takes the slope as 0, and so does a left
    difference. A symmetric difference would report the average, 1/2, which is
    what no real attribution-patching run ever sees.
    """
    hi = metric(forward(x, {node: at}), kind)
    lo = metric(forward(x, {node: at - eps}), kind)
    return (hi - lo) / eps


def mediation(treatment: str, mediator: str, base: Sequence[float], changed: float, kind: str = "prob") -> Dict:
    """Total, natural direct and natural indirect effect of one input on the output through one node

    TE  = Y(x')          - Y(x)
    NDE = Y(x', M(x))    - Y(x)     change the treatment, hold the mediator where it was
    NIE = Y(x, M(x'))    - Y(x)     keep the treatment, move only the mediator
    On a nonlinear model NDE + NIE need not equal TE; the gap is the interaction.
    """
    i = INPUTS.index(treatment)
    x0 = list(base)
    x1 = list(base)
    x1[i] = changed
    y0, y1 = forward(x0), forward(x1)
    m0, m1 = y0[mediator], y1[mediator]
    te = metric(y1, kind) - metric(y0, kind)
    nde = metric(forward(x1, {mediator: m0}), kind) - metric(y0, kind)
    nie = metric(forward(x0, {mediator: m1}), kind) - metric(y0, kind)
    return {"te": te, "nde": nde, "nie": nie, "interaction": te - nde - nie,
            "mediator_before": m0, "mediator_after": m1, "y_before": metric(y0, kind), "y_after": metric(y1, kind)}


def truth_table() -> List[Dict]:
    rows = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                v = forward([a, b, c])
                rows.append({"a": a, "b": b, "c": c, "prob": v["prob"], "target": int((a and b) or c)})
    return rows
