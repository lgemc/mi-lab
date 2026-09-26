from typing import Dict, List, Optional, Sequence

"""
Every token readout run on WorkspaceBench, ranked against the others on the same model.

A row is one method on one model, pooled over the banks it ran. Pooled by
counting items, never by averaging per-bank rates: a bank of 98 and a bank of
100 are not the same weight, and a clean hit rate is a fraction of the *clean*
items, which is a different count in every bank.

Methods are ranked only against methods on the same model and the same banks.
A run that covered fewer banks than the most complete one on its model is
marked incomplete and left out of the ranking rather than ranked on a subset
that happens to be easy -- the 2-bank smoke run on the 1.7B would otherwise
be read as a result.

Every number here was scored by the same `lens.score_bank`, so the table
compares readouts and nothing else. It is not WorkspaceBench's official
score: that is an LLM judge at each item's own layer and position, and this is
a string match over the top 10 tokens at any layer.

A common pipe could be: results/wsbench/*.json | rows | rank | markdown
"""


def pooled(state: Dict) -> Dict:
    """One results file reduced to the numbers a leaderboard row carries"""
    banks = state["banks"]
    every = [item for held in banks.values() for item in held["items"]]
    clean = [item for item in every if not item["leaked"]]
    items, leaked = len(every), sum(item["leaked"] for item in every)
    hits = sum(bool(item["hit_layers"]) for item in every)
    clean_hits = sum(bool(item["hit_layers"]) for item in clean)
    layers = sorted(item["best_layer"] for item in every if item.get("best_layer") is not None)
    return {
        "method": state.get("method", "logit-lens"),
        "config": state["config"],
        "lens": (state.get("lens") or {}).get("source", ""),
        "banks": sorted(banks),
        "items": items,
        "hit_rate": hits / items if items else 0.0,
        "clean_hit_rate": clean_hits / len(clean) if clean else None,
        "leak_rate": leaked / items if items else 0.0,
        "median_first_hit_layer": layers[len(layers) // 2] if layers else None,
        "per_bank": {family: {"items": held["summary"]["items"], "hit_rate": held["summary"]["hit_rate"],
                              "clean_hit_rate": held["summary"]["clean_hit_rate"]}
                     for family, held in banks.items()},
    }


def rank(rows: Sequence[Dict], expected: Optional[Sequence[str]] = None) -> List[Dict]:
    """Rows grouped by model, complete ones ranked by clean hit rate, incomplete ones after and unranked

    Clean hit rate is the headline: on a leaked item the readout cannot be
    told from the model having said the word (see `lens.report`).

    Complete means every bank in `expected` -- the benchmark's own list --
    and not merely as many as the widest run so far, or a model with one
    partial run would rank it first.
    """
    ranked: List[Dict] = []
    for config in sorted({row["config"] for row in rows}):
        group = [dict(row) for row in rows if row["config"] == config]
        if expected is not None:
            full = set(expected)
        else:
            widest = max(len(row["banks"]) for row in group)
            full = {bank for row in group if len(row["banks"]) == widest for bank in row["banks"]}
        complete = [row for row in group if set(row["banks"]) == full]
        incomplete = [row for row in group if set(row["banks"]) != full]
        complete.sort(key=lambda row: -(row["clean_hit_rate"] or 0.0))
        for position, row in enumerate(complete, 1):
            row["rank"], row["complete"] = position, True
        for row in incomplete:
            row["rank"], row["complete"] = None, False
        ranked.extend(complete + incomplete)
    return ranked


def percent(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def markdown(ranked: Sequence[Dict]) -> str:
    """The leaderboard as a page: the ranking per model, then every bank side by side"""
    lines = ["# WorkspaceBench token readouts", "",
             "String match of the target in the top 10 tokens at any layer, on the single-token banks. "
             "**Clean** drops items whose answer already contains the target and is the ranked number. "
             "Not the benchmark's official LLM-judged score.", ""]
    for config in dict.fromkeys(row["config"] for row in ranked):
        group = [row for row in ranked if row["config"] == config]
        lines += [f"## {config}", "",
                  "| rank | method | clean hit | hit | leaked | first hit layer | items | banks | lens |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in group:
            lines.append(f"| {row['rank'] or 'incomplete'} | {row['method']} | {percent(row['clean_hit_rate'])} | "
                         f"{percent(row['hit_rate'])} | {percent(row['leak_rate'])} | "
                         f"{row['median_first_hit_layer'] if row['median_first_hit_layer'] is not None else '—'} | "
                         f"{row['items']} | {len(row['banks'])} | {row['lens'] or '—'} |")
        complete = [row for row in group if row["complete"]]
        if len(complete) > 1:
            methods = [row["method"] for row in complete]
            lines += ["", "Clean hit rate per bank:", "",
                      "| bank | items | " + " | ".join(methods) + " |",
                      "|---|---|" + "---|" * len(methods)]
            for bank in complete[0]["banks"]:
                values = [row["per_bank"][bank]["clean_hit_rate"] for row in complete]
                best = max((value for value in values if value is not None), default=None)
                cells = [f"**{percent(value)}**" if value is not None and value == best else percent(value)
                         for value in values]
                lines.append(f"| {bank} | {complete[0]['per_bank'][bank]['items']} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
