"""Pearl's machinery on small DAGs: d-separation, the adjustment criteria, the rules of do-calculus

The graph is the whole input. A latent confounder is an ordinary node marked
`latent`, so "X <-> Y" is drawn as X <- U -> Y and every criterion below simply
refuses to condition on it -- which is what "unobserved" means operationally.

Everything is enumeration over simple paths. That is exponential in general and
instant on the dozen-node graphs a lesson draws, and it has the one property a
teaching tool needs: every verdict comes with the paths and the triple on each
path that decided it, so the page can show *why*, not just *whether*.
"""

from dataclasses import dataclass, field
from itertools import combinations, pairwise
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Edge = Tuple[str, str]


@dataclass
class DAG:
    nodes: List[str]
    edges: List[Edge]
    latent: Set[str] = field(default_factory=set)

    def __post_init__(self):
        known = set(self.nodes)
        for a, b in self.edges:
            if a not in known or b not in known:
                raise ValueError(f"edge {a}->{b} names a node that is not in the graph")
        if self._has_cycle():
            raise ValueError("the graph has a directed cycle, so it is not a DAG")

    def parents(self, n: str) -> Set[str]:
        return {a for a, b in self.edges if b == n}

    def children(self, n: str) -> Set[str]:
        return {b for a, b in self.edges if a == n}

    def descendants(self, n: str) -> Set[str]:
        """Strict descendants: everything reachable along directed edges, not n itself"""
        seen, stack = set(), [n]
        while stack:
            for c in self.children(stack.pop()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def ancestors(self, nodes: Iterable[str]) -> Set[str]:
        """Ancestors of a set, the set itself included"""
        seen, stack = set(nodes), list(nodes)
        while stack:
            for p in self.parents(stack.pop()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def _has_cycle(self) -> bool:
        state: Dict[str, int] = {}

        def visit(n: str) -> bool:
            state[n] = 1
            for c in self.children(n):
                if state.get(c) == 1 or (c not in state and visit(c)):
                    return True
            state[n] = 2
            return False

        return any(n not in state and visit(n) for n in self.nodes)

    def without_incoming(self, targets: Iterable[str]) -> "DAG":
        """G with every arrow INTO the targets removed: the graph of do(targets)"""
        t = set(targets)
        return DAG(self.nodes, [(a, b) for a, b in self.edges if b not in t], set(self.latent))

    def without_outgoing(self, sources: Iterable[str]) -> "DAG":
        """G with every arrow OUT of the sources removed"""
        s = set(sources)
        return DAG(self.nodes, [(a, b) for a, b in self.edges if a not in s], set(self.latent))

    def simple_paths(self, x: str, y: str, limit: int = 200) -> List[List[str]]:
        """Every simple path between x and y in the skeleton, shortest first"""
        adjacency: Dict[str, Set[str]] = {n: set() for n in self.nodes}
        for a, b in self.edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        out: List[List[str]] = []

        def walk(path: List[str]):
            if len(out) >= limit:
                return
            last = path[-1]
            if last == y:
                out.append(list(path))
                return
            for nxt in sorted(adjacency[last]):
                if nxt not in path:
                    path.append(nxt)
                    walk(path)
                    path.pop()

        walk([x])
        return sorted(out, key=len)

    def arrow(self, a: str, b: str) -> bool:
        return (a, b) in self.edges


@dataclass
class Triple:
    node: str
    kind: str  # chain | fork | collider
    blocked: bool
    reason: str


def judge_path(g: DAG, path: Sequence[str], given: Set[str]) -> Tuple[bool, List[Triple]]:
    """Whether a path is open given a conditioning set, and the verdict at every middle node

    The two rules, and nothing else: a chain or fork is blocked by conditioning
    on its middle node; a collider is blocked *unless* its middle node or one of
    its descendants is conditioned on.
    """
    triples: List[Triple] = []
    for a, b, c in zip(path, path[1:], path[2:], strict=False):
        if g.arrow(a, b) and g.arrow(c, b):
            opened = b in given or bool(g.descendants(b) & given)
            if b in given:
                reason = f"{b} is a collider and is conditioned on, which opens it"
            elif opened:
                seen = sorted(g.descendants(b) & given)
                reason = f"{b} is a collider and its descendant {', '.join(seen)} is conditioned on, which opens it"
            else:
                reason = f"{b} is a collider and nothing at or below it is conditioned on, so it blocks"
            triples.append(Triple(b, "collider", not opened, reason))
        else:
            kind = "fork" if (g.arrow(b, a) and g.arrow(b, c)) else "chain"
            blocked = b in given
            reason = (f"{b} is a {kind} and is conditioned on, so it blocks" if blocked
                      else f"{b} is a {kind} and is not conditioned on, so information flows through")
            triples.append(Triple(b, kind, blocked, reason))
    return not any(t.blocked for t in triples), triples


def d_separated(g: DAG, xs: Iterable[str], ys: Iterable[str], given: Iterable[str]) -> bool:
    z = set(given)
    return not any(judge_path(g, p, z)[0] for x in xs for y in ys for p in g.simple_paths(x, y))


def is_backdoor_set(g: DAG, x: str, y: str, z: Set[str]) -> bool:
    """Pearl's back-door criterion: no descendant of x in z, and z blocks every path into x"""
    if z & (g.descendants(x) | g.latent | {x, y}):
        return False
    return not any(len(p) > 1 and g.arrow(p[1], x) and judge_path(g, p, z)[0] for p in g.simple_paths(x, y))


def is_frontdoor_set(g: DAG, x: str, y: str, m: Set[str]) -> bool:
    """Pearl's front-door criterion for a mediator set m

    (i) m intercepts every directed path x -> y; (ii) no unblocked back-door
    path from x to m; (iii) every back-door path from m to y is blocked by x.
    """
    if not m or m & (g.latent | {x, y}):
        return False
    for p in g.simple_paths(x, y):
        directed = all(g.arrow(a, b) for a, b in pairwise(p))
        if directed and not set(p[1:-1]) & m:
            return False
    for mi in m:
        if not is_backdoor_set(g, x, mi, set()):
            return False
        if not is_backdoor_set(g, mi, y, {x}):
            return False
    return True


def minimal_sets(g: DAG, x: str, y: str, test, max_size: int = 3) -> List[List[str]]:
    """Every set passing `test` that has no passing proper subset, up to max_size"""
    candidates = [n for n in g.nodes if n not in (x, y) and n not in g.latent]
    found: List[FrozenSet[str]] = []
    for size in range(0, max_size + 1):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            if any(f <= s for f in found):
                continue
            if test(g, x, y, set(s)):
                found.append(s)
    return [sorted(s) for s in found]


def analyze(g: DAG, x: str, y: str, given: Sequence[str]) -> Dict:
    """Every path x..y with its verdict, whether x and y are d-separated, and the adjustment sets"""
    z = set(given)
    paths = []
    for p in g.simple_paths(x, y):
        open_, triples = judge_path(g, p, z)
        arrows = ["->" if g.arrow(a, b) else "<-" for a, b in pairwise(p)]
        paths.append({
            "nodes": p,
            "arrows": arrows,
            "open": open_,
            "backdoor": len(p) > 1 and g.arrow(p[1], x),
            "directed": all(a == "->" for a in arrows),
            "triples": [t.__dict__ for t in triples],
        })
    return {
        "paths": paths,
        "d_separated": not any(p["open"] for p in paths),
        "backdoor_sets": minimal_sets(g, x, y, is_backdoor_set),
        "frontdoor_sets": minimal_sets(g, x, y, is_frontdoor_set, max_size=2)[:6],
        "given_is_backdoor": is_backdoor_set(g, x, y, z),
        "descendants_of_x": sorted(g.descendants(x)),
    }


def do_rule(g: DAG, rule: int, y: Sequence[str], x: Sequence[str], z: Sequence[str], w: Sequence[str]) -> Dict:
    """Check the side condition of one rule of do-calculus, returning the graph it was checked in

    Rule 1 (insert/delete an observation):  P(y|do(x),z,w) = P(y|do(x),w)   if (Y ⊥ Z | X,W) in G_{X̄}
    Rule 2 (swap an action for an observation): P(y|do(x),do(z),w) = P(y|do(x),z,w)
                                                if (Y ⊥ Z | X,W) in G_{X̄ Z̲}
    Rule 3 (insert/delete an action):       P(y|do(x),do(z),w) = P(y|do(x),w)
                                                if (Y ⊥ Z | X,W) in G_{X̄ Z(W)̄}, where Z(W) is the
                                                part of Z that is not an ancestor of W in G_{X̄}
    """
    xs, zs, ws, ys = set(x), set(z), set(w), set(y)
    base = g.without_incoming(xs)
    if rule == 1:
        h, label = base, "G with arrows into X removed"
    elif rule == 2:
        h, label = base.without_outgoing(zs), "G with arrows into X and out of Z removed"
    elif rule == 3:
        z_w = zs - base.ancestors(ws) if ws else zs
        h, label = base.without_incoming(z_w), "G with arrows into X and into Z(W) removed"
    else:
        raise ValueError("do-calculus has three rules")
    given = xs | ws
    witnesses = []
    for yi in sorted(ys):
        for zi in sorted(zs):
            for p in h.simple_paths(zi, yi):
                open_, triples = judge_path(h, p, given)
                if open_:
                    witnesses.append({"nodes": p, "triples": [t.__dict__ for t in triples]})
    return {"holds": not witnesses, "graph": label, "edges": [list(e) for e in h.edges],
            "open_paths": witnesses[:5]}


# ----------------------------------------------------------------- front door, with numbers


def _bern(p: float, v: int) -> float:
    return p if v == 1 else 1 - p


def frontdoor(pu: float, px_u: Sequence[float], pm_x: Sequence[float], py_mu: Sequence[Sequence[float]]) -> Dict:
    """The smoking / tar / cancer model, exactly: U -> X, U -> Y, X -> M -> Y, with U hidden

    Arguments are the conditional probability tables: P(U=1); P(X=1|U=u) for u
    in 0,1; P(M=1|X=x); P(Y=1|M=m,U=u) indexed [m][u]. Returns the naive
    conditional, the truth (which needs U and so no observer has it), and the
    front-door estimate (which uses only X, M, Y), with every table the manual
    calculation passes through.
    """
    joint = np.zeros((2, 2, 2, 2))  # u x m y
    for u in (0, 1):
        for xv in (0, 1):
            for m in (0, 1):
                for yv in (0, 1):
                    joint[u, xv, m, yv] = (_bern(pu, u) * _bern(px_u[u], xv) * _bern(pm_x[xv], m)
                                           * _bern(py_mu[m][u], yv))
    obs = joint.sum(axis=0)  # x m y
    p_x = obs.sum(axis=(1, 2))
    p_m_given_x = obs.sum(axis=2) / p_x[:, None]
    p_y_given_xm = obs[:, :, 1] / obs.sum(axis=2)
    naive = [float(obs[xv, :, 1].sum() / p_x[xv]) for xv in (0, 1)]
    truth = [float(sum(_bern(pu, u) * _bern(pm_x[xv], m) * py_mu[m][u] for u in (0, 1) for m in (0, 1)))
             for xv in (0, 1)]
    inner = [[float(sum(p_y_given_xm[x2, m] * p_x[x2] for x2 in (0, 1))) for m in (0, 1)]]
    estimate = [float(sum(p_m_given_x[xv, m] * inner[0][m] for m in (0, 1))) for xv in (0, 1)]
    return {
        "observed_joint": obs.round(6).tolist(),
        "p_x": p_x.round(6).tolist(),
        "p_m_given_x": p_m_given_x.round(6).tolist(),
        "p_y_given_xm": p_y_given_xm.round(6).tolist(),
        "inner_sum": inner[0],
        "naive": naive,
        "truth": truth,
        "frontdoor": estimate,
        "effects": {
            "naive": naive[1] - naive[0],
            "truth": truth[1] - truth[0],
            "frontdoor": estimate[1] - estimate[0],
        },
    }


def parse_graph(nodes: Sequence[Dict], edges: Sequence[Sequence[str]]) -> DAG:
    return DAG([n["id"] for n in nodes], [(a, b) for a, b in edges], {n["id"] for n in nodes if n.get("latent")})


def optional_list(v: Optional[Sequence[str]]) -> List[str]:
    return list(v or [])
