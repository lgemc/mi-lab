"""An edge circuit reduced to the structural claim: which components, and which paths between them.

`sheaf_prune --edges-only` finishes with 3,720 open edges out of 14,140 and a
held-out score. That is a number and a list of tuples, not a circuit anyone can
read. `gates.summary` does the equivalent job for a weight mask -- reduce it to
the `mlp:L` / `head:L:H` vocabulary the rest of this repo argues in -- and this
is that function for the other kind, which had none: the GPT-2 IOI circuit was
read once out of a throwaway script and never written down.

The reduction has to say two different things, and only the first has a weight
analogue:

  which components   a head is in the circuit if any path out of it survived.
                     That is directly comparable to phase 1b's
                     `mlp:21,23,24,26 + heads:23` and to a published head list,
                     which is what makes a circuit checkable against prior work.
  which paths        the half a component list cannot express. A source writes
                     into the residual stream once and every later component
                     reads the same sum, so "kept for `attn:10`, cut for
                     `mlp:14`" is a fact about an edge and is invisible in any
                     per-component summary. `split` counts exactly those, and it
                     is the number that says whether an edge circuit was worth
                     finding: if it is near zero the circuit is a component
                     selection wearing a graph's clothes.

Degrees are reported against *available*, never against the model. A source at
layer 0 can reach 55 destinations and one at layer 27 can reach 1, so an
out-degree of 4 means opposite things at the two ends and dividing by the total
edge count would hide it.

A common pipe could be: edges_open | reduce | components + split | report
"""

from collections import defaultdict
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

Edge = Tuple[str, str]


class WiringError(ValueError):
    """Raised when an edge set cannot be read as a circuit for the model in hand"""


def parse_source(source: str) -> Tuple[str, Optional[int], Optional[int]]:
    """`head:9:6` -> ('head', 9, 6); `mlp:3` -> ('mlp', 3, None); `embed` -> ('embed', None, None)"""
    if source == "embed":
        return "embed", None, None
    parts = source.split(":")
    if parts[0] == "head" and len(parts) == 3:
        return "head", int(parts[1]), int(parts[2])
    if parts[0] in ("mlp", "bias") and len(parts) == 2:
        return parts[0], int(parts[1]), None
    raise WiringError(f"'{source}' is not a source this vocabulary knows")


def component_of(source: str) -> str:
    """The name this repo's other methods would use for the thing that wrote

    `bias:L` folds into `mlp:L`'s layer for reporting but keeps its own name --
    a bias is not a component anyone ablates, and silently merging it would
    inflate an MLP's degree with edges that are not the MLP's.
    """
    return source


def reduce(edges_open: Sequence[Edge], every: Sequence[Edge]) -> Dict[str, object]:
    """The circuit as components and degrees, from the open edges and the graph they live in

    `every` is the full edge list the model admits (`adapter.edges()`), and it
    is required rather than derived: the denominator of every degree is how
    many destinations that source *could* have reached, and an open-edge list
    on its own does not know what it left out.
    """
    open_set = {tuple(edge) for edge in edges_open}
    unknown = open_set - {tuple(edge) for edge in every}
    if unknown:
        raise WiringError(
            f"{len(unknown)} open edges are not in this model's graph, e.g. {sorted(unknown)[0]}. "
            "An edge names a layer and a head, so a circuit from another architecture "
            "cannot be read here."
        )

    out_total: Dict[str, int] = defaultdict(int)
    out_kept: Dict[str, int] = defaultdict(int)
    in_total: Dict[str, int] = defaultdict(int)
    in_kept: Dict[str, int] = defaultdict(int)
    readers: Dict[str, List[str]] = defaultdict(list)
    by_layer: Dict[int, int] = defaultdict(int)

    for edge in every:
        source, destination = tuple(edge)
        kept = (source, destination) in open_set
        out_total[source] += 1
        in_total[destination] += 1
        if kept:
            out_kept[source] += 1
            in_kept[destination] += 1
            readers[source].append(destination)
            by_layer[int(destination.split(":")[1])] += 1

    # A source is *split* when the circuit keeps it for some readers and cuts it
    # for others. That is the claim no weight mask can make, so it is counted
    # first and reported before the component list.
    split = [s for s in out_total if 0 < out_kept[s] < out_total[s]]
    all_kept = [s for s in out_total if out_kept[s] == out_total[s]]
    all_cut = [s for s in out_total if out_kept[s] == 0]

    components = []
    for source in sorted(out_total, key=lambda s: (-out_kept[s], s)):
        if not out_kept[source]:
            continue
        kind, layer, head = parse_source(source)
        components.append({
            "component": component_of(source),
            "kind": kind, "layer": layer, "head": head,
            "out_kept": out_kept[source], "out_available": out_total[source],
            "reach": round(out_kept[source] / out_total[source], 4),
            "split": source in set(split),
            "readers": sorted(readers[source]),
        })

    destinations = [
        {"destination": d, "kind": d.split(":")[0], "layer": int(d.split(":")[1]),
         "in_kept": in_kept[d], "in_available": in_total[d],
         "reach": round(in_kept[d] / in_total[d], 4) if in_total[d] else 0.0}
        for d in sorted(in_total, key=lambda d: (int(d.split(":")[1]), d))
    ]

    heads = [row for row in components if row["kind"] == "head"]
    mlps = [row for row in components if row["kind"] == "mlp"]
    return {
        "n_edges": len(every),
        "n_edges_open": len(open_set),
        "edge_density": round(len(open_set) / len(every), 6) if every else 0.0,
        "n_sources": len(out_total),
        "sources_all_kept": len(all_kept),
        "sources_all_cut": len(all_cut),
        "sources_split": len(split),
        # the component-level claim, comparable to every other method here
        "heads_in_circuit": [f"head:{r['layer']}:{r['head']}" for r in heads],
        "mlps_in_circuit": [f"mlp:{r['layer']}" for r in mlps],
        "components": components,
        "destinations": destinations,
        "by_layer": {str(layer): by_layer[layer] for layer in sorted(by_layer)},
    }


def against(heads_in_circuit: Sequence[str], known: Dict[str, str],
            n_heads_total: int) -> Dict[str, object]:
    """Overlap with a published head list, tested rather than eyeballed

    A count of matches means nothing on its own: keeping 40 of 144 heads hits a
    26-head reference 7.2 times on average. Comparing `found` to that mean and
    calling the difference enrichment is how 9-against-7.2 becomes a
    replication, so this reports the exact tail probability instead.

    Selecting k heads out of N without replacement, R of which are in the
    reference, the overlap is hypergeometric, and `p_value` is P(X >= found)
    under it. Computed with `math.comb` rather than estimated: the numbers here
    are small, the exact answer is three lines, and a permutation estimate
    would put sampling noise on top of the thing being measured.

    It is one-sided and uncorrected, and it tests only "more overlap than a
    random head set of the same size" -- not that the circuit is right, and not
    that the heads do what the reference says they do.
    """
    heads = {h.split(":", 1)[1].replace(":", ".") for h in heads_in_circuit}
    hits = sorted((h, role) for h, role in known.items() if h in heads)
    kept, reference, found = len(heads), len(known), len(hits)
    # keeping k of N heads hits an R-head reference k*R/N times on average
    expected = kept * reference / n_heads_total if n_heads_total else 0.0
    p_value = None
    if n_heads_total and kept <= n_heads_total and reference <= n_heads_total:
        total = comb(n_heads_total, kept)
        p_value = sum(
            comb(reference, i) * comb(n_heads_total - reference, kept - i)
            for i in range(found, min(kept, reference) + 1)
            if kept - i <= n_heads_total - reference
        ) / total
    return {
        "kept": kept,
        "of_total": n_heads_total,
        "reference": reference,
        "found": found,
        "expected_by_chance": round(expected, 2),
        # P(overlap >= found) for a random head set of the same size, exact
        "p_value": None if p_value is None else round(p_value, 4),
        "matches": [{"head": h, "role": role} for h, role in hits],
        "missing": sorted(set(known) - heads),
    }
