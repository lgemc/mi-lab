"""The HTTP face of the atlas: one route per lesson question, and the built front end at /

Every body is a pydantic model with bounds, because this is on the public
internet and each trainer's cost grows with its arguments: a slider cannot ask
for a million steps. Trainers are lru-cached by their arguments, so the second
visitor to drag a slider to the same place gets the answer for free.
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import causal, crosscoder, ioi, lens, mlp, probing, sae, superposition, transcoder

app = FastAPI(title="mi-lab atlas", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/api/health")
def health() -> Dict:
    return {"ok": True}


# ------------------------------------------------------------------------------ causality


class Node(BaseModel):
    id: str = Field(min_length=1, max_length=24)
    latent: bool = False


class GraphQuery(BaseModel):
    nodes: List[Node] = Field(min_length=2, max_length=14)
    edges: List[List[str]] = Field(max_length=40)
    x: str
    y: str
    given: List[str] = Field(default_factory=list, max_length=12)


def _graph(q) -> causal.DAG:
    try:
        return causal.parse_graph([n.model_dump() for n in q.nodes], q.edges)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/causal/analyze")
def causal_analyze(q: GraphQuery) -> Dict:
    g = _graph(q)
    if q.x not in g.nodes or q.y not in g.nodes or q.x == q.y:
        raise HTTPException(422, "x and y must be two different nodes of the graph")
    return causal.analyze(g, q.x, q.y, q.given)


class RuleQuery(BaseModel):
    nodes: List[Node] = Field(min_length=2, max_length=14)
    edges: List[List[str]] = Field(max_length=40)
    rule: Literal[1, 2, 3]
    y: List[str] = Field(min_length=1, max_length=4)
    x: List[str] = Field(default_factory=list, max_length=4)
    z: List[str] = Field(min_length=1, max_length=4)
    w: List[str] = Field(default_factory=list, max_length=4)


@app.post("/api/causal/rule")
def causal_rule(q: RuleQuery) -> Dict:
    return causal.do_rule(_graph(q), q.rule, q.y, q.x, q.z, q.w)


Prob = Field(ge=0.001, le=0.999)


class FrontdoorQuery(BaseModel):
    pu: float = Prob
    px_u: List[float] = Field(min_length=2, max_length=2)
    pm_x: List[float] = Field(min_length=2, max_length=2)
    py_mu: List[List[float]] = Field(min_length=2, max_length=2)


@app.post("/api/causal/frontdoor")
def causal_frontdoor(q: FrontdoorQuery) -> Dict:
    flat = [q.pu, *q.px_u, *q.pm_x, *(v for row in q.py_mu for v in row)]
    if any(not 0 < v < 1 for v in flat) or any(len(r) != 2 for r in q.py_mu):
        raise HTTPException(422, "every probability must be strictly between 0 and 1")
    return causal.frontdoor(q.pu, q.px_u, q.pm_x, q.py_mu)


# ------------------------------------------------------------------------ network as SCM


@app.get("/api/mlp/meta")
def mlp_meta() -> Dict:
    return {"inputs": mlp.INPUTS, "layer1": mlp.LAYER1, "layer2": mlp.LAYER2, "edges": mlp.edges(),
            "descriptions": mlp.DESCRIPTIONS, "truth_table": mlp.truth_table()}


Bit = Field(ge=0, le=1)


class MlpRun(BaseModel):
    x: List[float] = Field(min_length=3, max_length=3)
    set: Dict[str, float] = Field(default_factory=dict)


@app.post("/api/mlp/run")
def mlp_run(q: MlpRun) -> Dict:
    unknown = set(q.set) - set(mlp.NODES)
    if unknown:
        raise HTTPException(422, f"unknown nodes {sorted(unknown)}")
    return mlp.forward(q.x, {k: max(-5.0, min(5.0, v)) for k, v in q.set.items()})


class MlpPatch(BaseModel):
    clean: List[float] = Field(min_length=3, max_length=3)
    corrupt: List[float] = Field(min_length=3, max_length=3)
    metric: Literal["logit", "prob"] = "logit"


@app.post("/api/mlp/patch")
def mlp_patch(q: MlpPatch) -> Dict:
    return mlp.patching(q.clean, q.corrupt, q.metric)


class MlpMediation(BaseModel):
    treatment: Literal["a", "b", "c"] = "b"
    mediator: str = "n1"
    base: List[float] = Field(min_length=3, max_length=3)
    changed: float = 1.0
    metric: Literal["logit", "prob"] = "prob"


@app.post("/api/mlp/mediation")
def mlp_mediation(q: MlpMediation) -> Dict:
    if q.mediator not in mlp.LAYER1 + mlp.LAYER2:
        raise HTTPException(422, "the mediator must be a hidden neuron")
    return mlp.mediation(q.treatment, q.mediator, q.base, q.changed, q.metric)


# ---------------------------------------------------------------------------------- IOI

NameField = Field(pattern="^(" + "|".join(ioi.NAMES) + ")$")


class IoiBase(BaseModel):
    io: str = NameField
    s: str = NameField
    template: Literal["ABBA", "BABA"] = "ABBA"


def _check_names(q: IoiBase):
    if q.io == q.s:
        raise HTTPException(422, "the indirect object and the subject must be different names")


@app.get("/api/ioi/meta")
def ioi_meta() -> Dict:
    return ioi.meta()


class IoiRun(IoiBase):
    ablate: Dict[str, Literal["zero", "mean"]] = Field(default_factory=dict)
    corrupt: Optional[Literal["abc", "swap"]] = None


@app.post("/api/ioi/run")
def ioi_run(q: IoiRun) -> Dict:
    _check_names(q)
    if set(q.ablate) - set(ioi.HEAD_NAMES):
        raise HTTPException(422, "unknown head")
    return ioi.forward_view(q.io, q.s, q.template, q.ablate, q.corrupt)


class IoiPatch(IoiBase):
    corrupt: Literal["abc", "swap"] = "abc"
    direction: Literal["denoise", "noise"] = "denoise"


@app.post("/api/ioi/patch")
def ioi_patch(q: IoiPatch) -> Dict:
    _check_names(q)
    return ioi.patch_heads(q.io, q.s, q.template, q.corrupt, q.direction)


class IoiEdges(IoiBase):
    corrupt: Literal["abc", "swap"] = "abc"
    tau: float = Field(default=0.1, ge=0.0, le=5.0)


@app.post("/api/ioi/edges")
def ioi_edges(q: IoiEdges) -> Dict:
    _check_names(q)
    return ioi.edges(q.io, q.s, q.template, q.corrupt, q.tau)


# --------------------------------------------------------------------------------- lenses


@app.get("/api/lens/meta")
def lens_meta() -> Dict:
    return lens.meta()


class LensProbe(BaseModel):
    subject: str = Field(pattern="^(" + "|".join(lens.SUBJECTS) + ")$")
    patch_country: Optional[str] = Field(default=None, pattern="^(" + "|".join(lens.COUNTRIES) + ")$")
    patch_layer: int = Field(default=2, ge=1, le=6)
    two_hop: float = 0.0


@app.post("/api/lens/probe")
def lens_probe(q: LensProbe) -> Dict:
    if q.two_hop not in lens.CORPUS_MIXES:
        raise HTTPException(422, f"two_hop must be one of {list(lens.CORPUS_MIXES)}")
    return lens.probe(q.subject, q.patch_country, q.patch_layer, q.two_hop)


# ------------------------------------------------------------------- features & dictionaries


class Superposition(BaseModel):
    n: int = Field(default=5, ge=2, le=8)
    sparsity: float = Field(default=0.9, ge=0.0, le=0.99)
    decay: float = Field(default=0.9, ge=0.5, le=1.0)
    seed: int = Field(default=0, ge=0, le=9)


@app.post("/api/superposition")
def superposition_train(q: Superposition) -> Dict:
    return superposition.train(q.n, 2, round(q.sparsity, 2), round(q.decay, 2), seed=q.seed)


class Sae(BaseModel):
    k_true: int = Field(default=5, ge=2, le=8)
    latents: int = Field(default=5, ge=2, le=16)
    p: float = Field(default=0.15, ge=0.02, le=0.5)
    l1: float = Field(default=0.3, ge=0.0, le=5.0)
    mode: Literal["relu", "topk"] = "relu"
    topk: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default=0, ge=0, le=9)


@app.post("/api/sae")
def sae_train(q: Sae) -> Dict:
    return sae.train(q.k_true, q.latents, round(q.p, 2), round(q.l1, 3), q.mode, q.topk, seed=q.seed)


class SaeSweep(BaseModel):
    k_true: int = Field(default=5, ge=2, le=8)
    latents: int = Field(default=5, ge=2, le=16)
    p: float = Field(default=0.15, ge=0.02, le=0.5)
    mode: Literal["relu", "topk"] = "relu"


@app.post("/api/sae/sweep")
def sae_sweep(q: SaeSweep) -> Dict:
    return sae.sweep(q.k_true, q.latents, round(q.p, 2), q.mode)


class Transcoder(BaseModel):
    latents: int = Field(default=8, ge=2, le=16)
    l1: float = Field(default=0.1, ge=0.0, le=2.0)


@app.post("/api/transcoder")
def transcoder_train(q: Transcoder) -> Dict:
    return transcoder.public(transcoder.train(q.latents, round(q.l1, 3)))


class TranscoderGraph(Transcoder):
    features: List[float] = Field(min_length=5, max_length=5)


@app.post("/api/transcoder/graph")
def transcoder_graph(q: TranscoderGraph) -> Dict:
    return transcoder.attribution_graph([max(0.0, min(1.0, v)) for v in q.features], q.latents, round(q.l1, 3))


class Crosscoder(BaseModel):
    mode: Literal["l1", "batchtopk"] = "l1"
    latents: int = Field(default=16, ge=4, le=24)
    l1: float = Field(default=0.2, ge=0.0, le=2.0)
    k: int = Field(default=2, ge=1, le=6)
    drift: float = Field(default=0.2, ge=0.0, le=1.0)


@app.post("/api/crosscoder")
def crosscoder_train(q: Crosscoder) -> Dict:
    return crosscoder.train(q.mode, q.latents, round(q.l1, 3), q.k, round(q.drift, 2))


class Probe(BaseModel):
    correlation: float = Field(default=0.9, ge=0.0, le=1.0)
    spurious_scale: float = Field(default=1.5, ge=0.0, le=4.0)
    noise: float = Field(default=0.3, ge=0.05, le=1.0)
    strength: float = Field(default=1.0, ge=0.0, le=3.0)


@app.post("/api/probe")
def probe_study(q: Probe) -> Dict:
    return probing.study(round(q.correlation, 2), round(q.spurious_scale, 2), round(q.noise, 2),
                         round(q.strength, 2))


# ---------------------------------------------------------------------------- the front end

STATIC = Path(os.environ.get("ATLAS_STATIC", Path(__file__).resolve().parents[2] / "web" / "dist"))

if STATIC.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        """Any path that is not a file is a client-side route: hand back the app and let it route"""
        target = (STATIC / path).resolve()
        if path and target.is_file() and STATIC.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(STATIC / "index.html")
