# Interp Atlas

Interpretability, from Pearl's do-calculus to circuits, lenses, SAEs, transcoders and crosscoders,
taught with toy models you can poke. Live at **https://interp.atelier.run**.

Every model here is a toy on purpose: a graph with a probability table, or a network small enough
to hold in your head, wired by hand or trained inside a request. That makes each lesson's idea
*exactly* true in its model, so a learner can see what an instrument finds when the story is true,
and where it fails even then.

| Part | Lessons | Model behind it (`api/atlas/`) |
|---|---|---|
| Causality by hand | seeing vs doing, d-separation, front door + do-calculus, counterfactuals | `causal.py`: path enumeration, back-/front-door criteria, the three rules; front-door CPTs enumerated exactly |
| A network is a causal model | network as SCM, patching / attribution / mediation, probing vs steering | `mlp.py`: seven neurons computing (a∧b)∨c with a redundant neuron and a saturated output; `probing.py` |
| Circuits | IOI head by head, circuit discovery (patching, EAP, EAP-IG, ACDC), logit / J / tuned lens | `ioi.py`: a hand-written six-head attention-only transformer; `lens.py`: a two-hop residual network with a rotated workspace |
| Features and dictionaries | superposition, SAEs, transcoders + attribution graphs, crosscoders | `superposition.py`, `sae.py`, `transcoder.py`, `crosscoder.py`: numpy trainers, lru-cached |

## Layout

- `api/`: FastAPI + numpy, no torch, and nothing from the lab's `src/`. Every route body is bounded
  because the server is public. `tests/` pins the claim each lesson makes about its model: if a
  weight change breaks a sentence on a page, a test fails.
- `web/`: React + Vite + TypeScript, hand-rolled SVG charts, KaTeX. `src/lessons/registry.ts` is
  the table of contents; each lesson is one file.
- `Dockerfile`: one image. The web build is served by the API at `/`, and the API is at `/api`.

## Develop

```bash
cd atlas/api && uv sync && uv run uvicorn atlas.app:app --port 8010 --reload
cd atlas/web && npm install && npm run dev          # proxies /api to :8010
cd atlas/api && uv run python -m unittest tests.causal tests.models tests.app
cd atlas/web && npm run build                       # typecheck + bundle
```

## Deploy

Manifests and scripts are in `~/main/m/projects/k8s/atlas`: `10-build-image.sh` (build and import
into k3s containerd), `20-deploy.sh [--restart]`, `30-public-route.sh` (DNS record plus the
cloudflared rule for `interp.atelier.run`).

The atlas is a server rather than a run, so like `scripts/serve.py` it is not traced in MLflow:
nothing it computes is a result.
