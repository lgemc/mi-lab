"""Phase 1c, revived: attribution graphs over a pretrained cross-layer transcoder.

`docs/methodology.tex` lists Phase 1c as "SAE features (CONDITIONAL) --
replaced by edge pruning", shelved because training a dictionary for this
study's model was not affordable. It is affordable now, for one checkpoint and
only one: BluelightAI published a cross-layer transcoder for
Qwen3-1.7B-Base, and a CLT is the thing an attribution graph is made of
(Ameisen et al. 2025, `~/main/m/papers/circuit_tracing/methods.html`).

This is the third base, and that is the whole reason it exists. The same
question -- which parts of this model do the Spanish-to-English translation --
now has three answers on one checkpoint, produced by three different
decompositions:

  knockout    `phase1b_ablation`: ablate a component, regenerate, measure BLEU.
              The measurement, and the slowest.
  attribution `phase1b_attribution`: EAP over the same heads and MLPs. The
              same causal question, first-order, ~600x cheaper.
  graphs      here: features rather than components, edges between them from
              backward Jacobians with the attention patterns frozen.

What is *not* shared is the sign of the numbers or their units, so nothing in
this file compares them arithmetically. A graph is per-prompt and its nodes
are features; a sweep is per-corpus and its nodes are components. `summarise`
reports the two facts that survive that gap -- which layers the graph's
features live in, and how much of the graph is error nodes -- and the layer
profile is the only thing that lines up with the other two bases.

The error-node share is printed first and on purpose. Whatever the transcoder
fails to reconstruct enters the graph as an error node, error nodes have no
incoming edges, and so they are exactly where the explanation stops. This
release leaves ~23% of MLP output unexplained (`configs/qwen3-1.7b-base.yaml`
records it). A graph read without that number in view reads as more complete
than it is; the paper lists it first among its own limitations, under "dark
matter".

Run: uv run python -m scripts.phase1c_graphs qwen3-1.7b-base check
     uv run python -m scripts.phase1c_graphs qwen3-1.7b-base graph --sentence 0
     uv run python -m scripts.phase1c_graphs qwen3-1.7b-base summarise
"""

import sys
import time
from typing import Any, Dict, List, Optional

from src.core.config import load_config
from src.experiment import translation_study as study
from src.model.replacement import load_replacement as load_with_backend
from src.telemetry.observe import banner, duration, gpu, log, set_log_file, step
from src.telemetry.results import guard, load_state, result, save_state
from src.telemetry.results import root as results_root
from src.telemetry.tracking import tracked_main

ARTIFACTS = {
    "graph": "phase1c-attribution-graph-{index}.pt",
    "summary": "phase1c-graph-summary.json",
}
LOG = study.log_path("graphs")

# The paper's own defaults, restated because they are choices: output nodes
# cover 95% of the probability mass up to ten tokens, and a graph is pruned
# to the subgraph that carries the behaviour rather than shown whole (the
# edge count reaches millions on short prompts).
LOGIT_PROBABILITY = 0.95
MAX_LOGITS = 10


class GraphError(study.StudyError):
    """A checkpoint with no transcoder, or a graph asked for before one was built"""


def artifact(key: str, **fields) -> Any:
    return result(ARTIFACTS[key].format(**fields))


def transcoder_of(config: str):
    """The dictionary this config names, refused clearly when it names none

    A model without a transcoder is not a model this phase can run on, and it
    is worth saying so with the reason rather than failing inside a loader: a
    residual-stream SAE will not do, because the construction replaces the
    MLP and an SAE reconstructs an activation from itself.
    """
    cfg = load_config(config)
    if cfg.transcoder is None:
        raise GraphError(
            f"'{config}' names no transcoder, and an attribution graph is built out of one. "
            "Published cross-layer transcoders exist for very few checkpoints; "
            "configs/qwen3-1.7b-base.yaml is the one in this repo that has one"
        )
    if cfg.transcoder.kind != "cross-layer":
        log(f"note: '{config}' names a {cfg.transcoder.kind} transcoder; the paper's construction "
            "uses cross-layer, and a per-layer set is the weaker variant it compares against")
    return cfg


def load_replacement(cfg, backend: str = "transformerlens"):
    """The local replacement model: this checkpoint with its MLPs replaced by the transcoder

    The loading itself is `src/model/replacement.py`, which a second script now
    needs too. What stays here is this study's choice of arguments and the log
    lines, because `src/model/` sits below `telemetry` and cannot write them.
    """
    model, attempt = load_with_backend(
        cfg.hf_name,
        cfg.transcoder.release,
        dtype=cfg.dtype,
        backend=backend,
        note=lambda message: log(message),
        wrap=step,
    )
    log(f"replacement model ready on the {attempt} backend", indent=1)
    return model, attempt

def prompts(count: int = 1, size: Optional[int] = None) -> List[str]:
    """The study's own eval prompts, so a graph is about the task the rest of the repo measures

    Not a hand-written sentence. The few-shot form, the shots and the sources
    are the ones `phase1b_ablation` ablated under and `phase1b_attribution`
    scored, and a graph built on a different prompt is a graph of a different
    computation.
    """
    corpus = study.Corpus.load(size)
    return corpus.prompts[:count]


def stage_check(config: str, options: Dict[str, Any]) -> None:
    """Load the thing and run one prompt through it, before spending a graph on it

    The cheapest failure is the one that happens in the first minute. This
    stage exists because everything after it costs tens of gigabytes of
    resident weights, and a backend that cannot load this architecture should
    say so before that and not during it.
    """
    cfg = transcoder_of(config)
    log(f"transcoder: {cfg.transcoder.release} ({cfg.transcoder.kind}, "
        f"{cfg.transcoder.features_per_layer} features/layer, L0~{cfg.transcoder.l0}, "
        f"{cfg.transcoder.resident_gib} GiB resident)")
    log(f"unexplained variance: {cfg.transcoder.variance_unexplained:.0%} of MLP output arrives as "
        "error nodes, which have no incoming edges -- that is where the explanation stops")
    model, backend = load_replacement(cfg, options["backend"])
    text = prompts(1)[0]
    with step("one forward pass through the replacement model") as facts:
        # The paper's own smallest evaluation: does the replacement model still
        # predict what the model predicts. Here it is one prompt and a top
        # token rather than a rate over a corpus, because the question this
        # stage answers is "did any of this load", and a replacement model that
        # continues a translation prompt with an English word has answered it.
        top = _top_token(model, text)
        facts["backend"] = backend
        facts["prompt_tail"] = repr(text[-48:])
        facts["predicts"] = repr(top)
    log("loaded and predicting; `graph` is the next stage")


def _top_token(model, text: str) -> str:
    """The replacement model's most likely next token for a prompt, however the backend spells that"""
    import torch

    with torch.no_grad():
        output = model(text)
    logits = output.logits if hasattr(output, "logits") else output
    best = int(logits[0, -1].argmax())
    tokenizer = getattr(model, "tokenizer", None)
    return tokenizer.decode([best]) if tokenizer is not None else str(best)


def stage_graph(config: str, options: Dict[str, Any]) -> None:
    """Build and save an attribution graph for one of the study's translation prompts"""
    # The submodule and the function share a name, so `from
    # circuit_tracer.attribution import attribute` resolves to whichever won
    # the import race and is a module about half the time. The full path is
    # unambiguous.
    from circuit_tracer.attribution.attribute import attribute

    cfg = transcoder_of(config)
    model, backend = load_replacement(cfg, options["backend"])
    index = options["sentence"]
    text = prompts(index + 1)[index]
    log(f"prompt {index}: ...{text[-80:]!r}")
    started = time.time()
    with step(f"attributing prompt {index}") as facts:
        graph = attribute(
            text, model,
            max_n_logits=MAX_LOGITS,
            desired_logit_prob=LOGIT_PROBABILITY,
            batch_size=options["batch"],
            max_feature_nodes=options["max_features"],
            offload=options["offload"],
            verbose=True,
        )
        facts["gpu"] = gpu()
    path = artifact("graph", index=index)
    graph.to_pt(path)
    log(f"graph written to {path} in {duration(time.time() - started)}")
    save_state(artifact("summary"), {
        **load_state(artifact("summary"), {}),
        str(index): {
            "prompt": text, "backend": backend, "transcoder": cfg.transcoder.release,
            "seconds": round(time.time() - started, 1), "path": str(path),
            "logit_token_ids": [int(token) for token in graph.logit_token_ids],
            "variance_unexplained": cfg.transcoder.variance_unexplained,
        },
    })


def read_graph(path):
    """One saved graph, with its nodes counted by kind

    The adjacency matrix is ordered `[features, errors, embeddings, logits]`
    with rows as targets and columns as sources, which circuit-tracer
    documents and this function restates only as slice boundaries -- an
    off-by-one here would attribute a feature's influence to an error node and
    the number would still look plausible.
    """
    import torch
    from circuit_tracer.graph import Graph

    graph = Graph.from_pt(path)
    n_features = int(graph.active_features.shape[0])
    n_errors = int(graph.cfg.n_layers) * int(graph.n_pos)
    n_embed = int(graph.n_pos)
    n_logits = len(graph.logit_targets)
    bounds = {
        "feature": (0, n_features),
        "error": (n_features, n_features + n_errors),
        "embedding": (n_features + n_errors, n_features + n_errors + n_embed),
        "logit": (n_features + n_errors + n_embed, n_features + n_errors + n_embed + n_logits),
    }
    total = bounds["logit"][1]
    if total != graph.adjacency_matrix.shape[0]:
        raise GraphError(
            f"the graph's node bookkeeping does not add up: {n_features} features + {n_errors} errors "
            f"+ {n_embed} embeddings + {n_logits} logits = {total}, but the adjacency matrix is "
            f"{tuple(graph.adjacency_matrix.shape)}; the node order this reads by has changed"
        )
    return graph, bounds, torch


def node_influence(graph, bounds, torch):
    """Each node's influence on the output, the paper's way: over paths, not one hop

    `compute_node_influence` is circuit-tracer's own implementation of
    "Computing Node Influence" -- it normalises the adjacency matrix and
    iterates it against the logit weights, so a feature that reaches the
    output only through three other features is credited for it. The one-hop
    alternative (just the logit rows of the adjacency matrix) is a different
    and much worse measure, and it is worth naming because it is the one you
    get by accident: it credits early layers for having many active features
    and says nothing about whether their influence arrives anywhere.
    """
    from circuit_tracer.graph import compute_node_influence

    # The weights are a vector over *every* node, zero everywhere but the
    # logit nodes -- it is the left-hand side of a repeated matrix product,
    # not a list of logit probabilities. Handing it the four probabilities
    # raises on a shape mismatch, which is the good case; a vector of the
    # right length pointing at the wrong slice would not.
    start, stop = bounds["logit"]
    weights = torch.zeros(graph.adjacency_matrix.shape[0], dtype=torch.float32)
    probabilities = graph.logit_probabilities.float()
    weights[start:stop] = probabilities / (probabilities.sum() or 1.0)
    return compute_node_influence(graph.adjacency_matrix.float(), weights)


def influence_by_kind(graph, bounds, torch) -> Dict[str, float]:
    """What share of the influence on the output comes from each kind of node

    The number worth reading is `error`. Every error node is MLP output the
    transcoder could not reconstruct, error nodes have no incoming edges, and
    so their share is the fraction of the output this method explains by
    saying "and then something happened". The release quotes ~23% of variance
    unexplained globally; this is what that comes to on *this* prompt, which
    is the number that belongs beside a graph someone is about to read.
    """
    influence = node_influence(graph, bounds, torch).abs()
    mass = {kind: float(influence[start:stop].sum())
            for kind, (start, stop) in bounds.items() if kind != "logit"}
    total = sum(mass.values()) or 1.0
    return {kind: value / total for kind, value in mass.items()}


def layers_of_influence(graph, bounds, torch, top: int = 8) -> List[Dict[str, Any]]:
    """Where in the stack the graph's features live, weighted by influence on the output

    The one thing that survives the gap between this base and the other two.
    A feature is not a component and these numbers are not BLEU, but "which
    layers carry the translation" is a claim all three bases make, and phase
    1b makes it by naming a band.
    """
    feature_mass = node_influence(graph, bounds, torch).abs()[bounds["feature"][0] : bounds["feature"][1]]
    layers = graph.active_features[:, 0]
    per_layer = torch.zeros(int(graph.cfg.n_layers))
    per_layer.index_add_(0, layers.cpu(), feature_mass.float().cpu())
    share = per_layer / (per_layer.sum() or 1.0)
    order = sorted(range(len(share)), key=lambda index: -float(share[index]))[:top]
    return [{"layer": index, "share": round(float(share[index]), 4),
             "features": int((layers == index).sum())} for index in order]


def stage_summarise(config: str, options: Dict[str, Any]) -> None:
    """What the graphs built so far say, in the only terms the other two bases share

    Deliberately narrow. A feature is not a component and an edge here is not
    an edge in `phase1b_attribution`, so nothing is compared arithmetically.
    What is reported is where the explanation lives and where it stops.
    """
    summary = load_state(artifact("summary"), {})
    if not summary:
        raise GraphError(f"no graphs built yet; run the graph stage first (looked in {artifact('summary')})")
    for index, record in sorted(summary.items()):
        graph, bounds, torch = read_graph(record["path"])
        kinds = influence_by_kind(graph, bounds, torch)
        log(f"prompt {index}: {bounds['feature'][1]} features, "
            f"{bounds['error'][1] - bounds['error'][0]} error, "
            f"{bounds['embedding'][1] - bounds['embedding'][0]} embedding, "
            f"{bounds['logit'][1] - bounds['logit'][0]} logit nodes "
            f"over {graph.n_pos} positions ({record['seconds']}s on {record['backend']})")
        log(f"influence on the output: features {kinds['feature']:.1%}, "
            f"error {kinds['error']:.1%}, embedding {kinds['embedding']:.1%}", indent=1)
        log(f"the error share is where the explanation stops -- the release quotes "
            f"{record['variance_unexplained']:.0%} unexplained variance overall", indent=1)
        for row in layers_of_influence(graph, bounds, torch):
            log(f"layer {row['layer']:>2}: {row['share']:6.1%} of feature influence, "
                f"{row['features']} active features", indent=2)


STAGES = {"check": stage_check, "graph": stage_graph, "summarise": stage_summarise}


def parse(argv: List[str]) -> Dict[str, Any]:
    positional = [word for word in argv if not word.startswith("--")]
    flags = {}
    for index, word in enumerate(argv):
        if word.startswith("--"):
            following = argv[index + 1] if index + 1 < len(argv) else "1"
            flags[word[2:]] = following if not following.startswith("--") else "1"
    return {
        "config": positional[0] if positional else "qwen3-1.7b-base",
        "stage": positional[1] if len(positional) > 1 else "check",
        "backend": flags.get("backend", "auto"),
        "sentence": int(flags.get("sentence", 0)),
        "batch": int(flags.get("batch", 128)),
        "max_features": int(flags["max-features"]) if "max-features" in flags else None,
        "offload": flags.get("offload") or None,
    }


def main(argv: List[str]) -> int:
    options = parse(argv)
    if options["stage"] not in STAGES:
        log(f"unknown stage '{options['stage']}'; stages are {', '.join(STAGES)}")
        return 2
    set_log_file(LOG)
    guard(options["config"])
    banner(f"phase 1c attribution graphs: {options['stage']}", {
        "config": options["config"],
        "backend": options["backend"],
        "sentence": options["sentence"],
        "offload": options["offload"],
    })
    started = time.time()
    STAGES[options["stage"]](options["config"], options)
    log(f"stage '{options['stage']}' done in {duration(time.time() - started)}")
    return 0


if __name__ == "__main__":
    sys.exit(tracked_main(lambda: main(sys.argv[1:]), "mi-lab-graphs", outputs=[results_root()]))
