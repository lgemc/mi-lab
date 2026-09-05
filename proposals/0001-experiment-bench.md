# **Proposal: the bench — a control plane for experiments**

| Field          | Value                                                                 |
| :------------- | :-------------------------------------------------------------------- |
| **Status**     | Proposed — not implemented                                             |
| **Date**       | September 2026                                                         |
| **Reference**  | new package `src/bench/`, new CLI group `mi-lab bench`, deployed from `~/main/m/projects/k8s/mi-lab-bench` |
| **Supersedes** | nothing. `scripts.pipeline` keeps working unchanged                    |

---

## I. The failure this exists to stop

A run in this repository is a process attached to a terminal. It dies when the terminal
dies, and the terminal dies for reasons that have nothing to do with the experiment.

Two in the last twenty-four hours, both recorded:

| When | What | Cost |
| :--- | :--- | :--- |
| 2026-09-04 17:39 | NVRM out-of-memory, box hard-crashed, reboot 17:46 | a finished 900-step run, noted in `pipelines/run/sheaf-ioi-1.7b-sweep.yaml` |
| 2026-09-05 03:09 | reboot | `prune-s0.1`, 21 minutes into a 1h56m step, at step 7 of 21 |

The second one is the shape of the problem. The sweep is resumable and resumed nothing,
because resuming is a human re-typing `uv run python -m scripts.pipeline
run=sheaf-ioi-1.7b-sweep` and no human was awake. The box came back at 03:09 with a free
GPU and sat idle for five hours and seventeen minutes. The state that would have restarted
it — six steps done, the seventh interrupted, fourteen to go — was on disk the whole time,
in `pipeline-state.json`, with nothing running that could read it.

So the missing piece is not resumability. `pipeline-state.json` is resumability, and
`telemetry/results.py` has been doing it per-script for longer than that. The missing piece
is **something alive that wants the work done** — a process whose job is to notice that a
task is pending and a worker is free, and that outlives every session that submits to it.

That is a control plane, and the shape of one is well known, so this proposal is mostly
about which of its parts are load-bearing here and which are ceremony.

### I.1 What is explicitly not being replaced

- **`telemetry/journal.py` stays exactly what it is.** It is deliberately not a server, and
  a flushed `metrics.jsonl` beside the run is still the answer to "what is it doing right
  now". The bench records the journal's *path*; it does not ingest its rows.
- **MLflow stays the mirror.** `telemetry/tracking.py` already answers "how does this run
  compare to the four before it". The bench records the run id and links to it.
- **`scripts.pipeline` stays runnable by hand.** A pipeline in a terminal is the right tool
  for a five-minute pipeline, and it is the fallback when the bench is the thing that is
  broken.
- **`pipeline-state.json` and every script's own progress file stay.** They are the second
  resume layer and they compose with the first — see §VI.2.

The bench adds a queue, a scheduler and a place to look. It does not become the new home of
anything that already has one.

---

## II. Object model

The k8s analogy is the user's and it is the right one, so the names follow it. What matters
is which k8s object each thing is *actually* like, because that decides the state machine.

| Kubernetes | Here | What it is |
| :--- | :--- | :--- |
| image in a registry | **Image** | a composed, content-addressed, immutable YAML config |
| tag | **Tag** | a mutable name pointing at one digest |
| Job | **Experiment** | one submitted image, expanded into tasks |
| Pod | **Task** | one executable unit — one pipeline step |
| kubelet | **Worker** | long-lived process that leases tasks and forks subprocesses |
| kube-scheduler | **Scheduler** | admits pending tasks against capacity, reclaims dead ones |
| apiserver + etcd | **API + Store** | the only writer of state |

### II.1 An Image is the *composed* config, not the file

`pipelines/run/sheaf-ioi-1.7b-sweep.yaml` is not a run. It is an input to Hydra composition,
and what determines the result is what comes out of that composition — the file plus
`pipelines/config.yaml`, plus the `sheaves/run/qwen-ioi-price.yaml` each `scripts.sheaf`
step names, plus the two overrides that step applies to it.

So **composition happens at build time and the resolved config is what is stored.** Three
things follow, and each is the point of doing it this way:

1. **A stored image cannot change under a queued run.** Editing
   `sheaves/run/qwen-ioi-price.yaml` after submitting does not alter what runs. Today it
   would, and nothing would say so.
2. **A misspelling fails in a second instead of in two hours.** `SheafSpec.from_mapping`
   already rejects unknown keys (invariant 3), and `run.sparsty=0.03` is exactly the error
   the sweep's own header complains about being "a two-hour surprise in an argv list". Build
   runs that check before anything is queued.
3. **The digest is a real identity.** Two files that compose to the same mapping are the
   same image and get the same digest — the discipline `spec_hash` already keeps (invariant
   4). A digest is `sha256` over the canonical JSON of the resolved mapping.

```
mi-lab bench build pipelines/run/sheaf-ioi-1.7b-sweep.yaml -t ioi-sweep:v1
  composed 21 steps, 7 sheaf specs checked
  sha256:7f3ac91e...  ->  ioi-sweep:v1
```

### II.2 The consequence: scripts need a door that takes a resolved spec

If the image holds the resolved config, the worker must be able to hand that config to the
script *without* re-composing. Today `scripts.sheaf` only takes Hydra overrides, which means
it would read `sheaves/` again at execution time and the stored image would be decorative.

So this proposal adds one door:

```bash
uv run python -m scripts.sheaf --spec /path/to/resolved.json
```

`src/experiment/sheaf.py` already states the intent — *"The argparse entrypoint stays and
builds the same record, so both doors open into one code path and one schema"* — and this is
a third door into the same `SheafSpec.from_mapping`. It is a small change and it is the one
that makes versioned images mean anything. Steps whose module is not `scripts.sheaf` need
nothing: their `args` are already literal.

This also decides what the worker image must contain. It does **not** need `sheaves/` or
`pipelines/`; those are build inputs, read by the CLI from a checkout. The image needs what
it already copies — `src`, `scripts`, `configs`, `data`.

### II.3 A Task is a step, and its dependencies are declared

A submitted pipeline image expands into one Task per step. `prune-s0.1` becomes its own
queued unit, which is what makes the reboot story work: the crash re-queues twenty-one
minutes of work, not fifteen hours of it.

Dependencies are **the declared order, as a linear chain, by default.** The sweep's steps
are not all sequentially dependent — `prune-s0.03` does not need `extract-s0.01` — but the
YAML declares an order and not a graph, and inferring independence from an ordered list is
guessing. A step may opt into the graph:

```yaml
- name: infer-s0.01
  needs: [prune-s0.01]
```

With one GPU this buys nothing today. It is written down now because the linear chain is a
*choice* and not a limit, and because the moment there are two workers the difference is
seven hours.

### II.4 States

```
Task:        Blocked ──> Pending ──> Assigned ──> Running ──> Succeeded
                            ▲                        │
                            │                        ├──> Failed
                            └──── requeue ───────────┴──> Lost
             (any) ──> Cancelled

Experiment:  Pending ──> Running ──> Succeeded | Failed | Cancelled
```

`Lost` is separate from `Failed` on purpose. `Failed` is a non-zero exit — the experiment
said no, and re-running it is likely to say no again. `Lost` is a lapsed lease — nobody
knows what happened, and re-running it is exactly right. Collapsing them means either
retrying real failures forever or never retrying a reboot. The 03:09 crash is `Lost`; a
`SheafError` on a bad target is `Failed`.

Experiment state is stored, not derived, so `ps` is one query. The scheduler owns it.

---

## III. The lease is the load-bearing part

Everything else here is bookkeeping. This is the mechanism that makes the system survive the
thing it was built for, so it gets its own section.

**A `running` boolean is a lie the moment the machine dies.** Reboot the box and the row
still says `Running`, forever, with no process behind it. Every naive job queue has this bug
and it is invisible until the first hard crash — which, on this box, is weekly.

So a task is not *marked* running, it is **claimed with a lease that expires**:

- A worker claims a task and receives a lease valid until `now() + lease_ttl` (default 60s).
- The worker heartbeats every `lease_ttl / 3`, renewing the lease and shipping progress.
- The scheduler moves any task whose `lease_expires_at < now()` to `Lost` and re-queues it.

Three details that are the difference between this working and appearing to work:

1. **The clock is the database's, never the worker's.** Expiry is evaluated in SQL against
   `now()`, and the lease is set in SQL. A worker with a skewed clock must not be able to
   hold a lease forever or lose one instantly.
2. **The lease carries a fencing token.** Each claim increments a monotonic
   `lease_epoch` on the task. A heartbeat presents its epoch; if the stored epoch has moved
   on, the worker has been reclaimed behind its back and **must kill its own subprocess and
   abandon the task.** Without this, a worker partitioned from the API keeps a two-hour
   prune running while a second worker starts the same prune into the same results
   directory — two processes writing `sheaf-ioi-mask.pt`, and the surviving file belongs to
   neither run. `telemetry/results.guard` does not catch this: both processes are the same
   config, which is the case it is designed to permit.
3. **A restarting worker does not adopt its old subprocess.** In a container the process
   tree dies with the container, so there is nothing to adopt. The worker starts clean and
   the task it was running is reclaimed by expiry like any other. Step-level granularity is
   what bounds the loss to one step.

---

## IV. Admission: one GPU, and the reason it is enforced

There is one GB10. A 2000-step 1.7B prune reports `~39.39 GiB peak` in its own banner and
runs for 1h56m. Two of them at once is the `NVRM out-of-memory` that hard-crashed this box
on 2026-09-04, and a queue that dispatches two pending prunes to two free slots would
reproduce it on purpose.

So admission is by declared capacity:

- A worker starts with `--slots N` (default **1**).
- A task declares `cost` (default **1**).
- The scheduler admits a task only if the worker's free slots cover its cost.

One worker with one slot serializes every prune, which is what the sweep wants and what the
hardware demands. This is deliberately cruder than k8s resource requests: modelling GPU
memory properly means knowing a step's peak before running it, `methods/gates.budget`
already prices a band *in the run*, and a second guess in the scheduler that disagrees with
it is worse than no guess. Slots first; a memory-aware admission filter is a later change
that does not move anything else.

---

## V. Components and where state may be written

```
        mi-lab bench (CLI)              on the host, from a checkout
               │  composes images, submits, reads
               ▼
        ┌──────────────┐        the ONLY holder of the DSN
        │  bench-api   │───────────────────┐
        └──────────────┘                   ▼
               ▲  ▲                  ┌───────────┐
               │  │                  │ postgres  │  (or sqlite)
               │  └──────────────────│  store    │
               │                     └───────────┘
        ┌──────┴───────┐
        │  scheduler   │   admits, expires leases, rolls up experiments
        └──────────────┘
               ▲
        ┌──────┴───────┐
        │   worker     │   claim → fork → heartbeat → report
        └──────────────┘
               │
               └─ subprocess:  python -m scripts.sheaf --spec ...
```

**Only the API process touches the store.** The scheduler and the worker are HTTP clients of
it, exactly as kube-scheduler and kubelet are clients of the apiserver. Two things follow
that are worth the indirection: a worker needs no database credentials, and every state
transition goes through one place that can validate it. At this scale — a handful of tasks
an hour — the single writer is not a bottleneck worth avoiding.

**No torch, and no model import, anywhere in `src/bench/`.** The same rule `telemetry` and
`experiment/run.py` keep, for the same reason plus one more: the API and scheduler must run
in a small pod with no GPU, and the worker must be able to report a failure it could not
have loaded a model to produce. The worker forks; the subprocess is the thing with torch.

### V.1 Where it sits in the dependency order

```
core        config, metrics                      imports nothing
telemetry   journal, tracking, observe, results  imports nothing
...
experiment  spec, run, runner, pipeline, sheaf   -> core, model, data, methods, share
bench       model, store, stores/, images,       -> core, telemetry, experiment
            expand, api, scheduler, worker
cli                                              -> everything
```

`bench` imports `experiment` for `pipeline.compose` / `sheaf.compose` and `SheafSpec` — and
only in `images.py`, the build path, which runs in the CLI. `api`, `scheduler` and `worker`
import neither. Nothing imports `bench`, exactly as nothing imports `ie`.

Hydra therefore stays out of the API and scheduler processes entirely.

---

## VI. The store, and the one query that has to be right

A `Store` protocol with two implementations, in the shape `model/adapter.py` and
`methods/discovery.py` already use — a protocol plus registered backends, selected by a DSN.

- `PostgresStore` — psycopg 3. `MI_LAB_BENCH_DSN=postgresql://...@pg-rw.ai-infra/bench`
- `SqliteStore` — stdlib `sqlite3`. The **default** when no DSN is set:
  `~/.mi-lab/bench.db`.

SQLite as the default is not a toy path. It is what makes `mi-lab bench` a daemon you can
run on the host with no cluster, no Postgres and no credentials — the fallback for exactly
the mornings when the cluster is what broke. Postgres is the backbone because it is already
there (`pg-rw.ai-infra`, CloudNativePG) and because it is the one that survives the node.

No ORM. Six tables and raw SQL, because this repo has no ORM and SQLAlchemy is a large new
surface for `images`, `tags`, `experiments`, `tasks`, `workers`, `logs`.

### VI.1 Claiming a task is the only interesting statement

Everything else is CRUD. This is the one place two dialects genuinely differ, and the one
place a race silently double-runs a two-hour job:

```sql
-- postgres: row-level lock, skip what another claimer holds
UPDATE tasks SET state = 'Assigned', worker_id = :w,
                 lease_epoch = lease_epoch + 1,
                 lease_expires_at = now() + :ttl
WHERE id = (SELECT id FROM tasks
            WHERE state = 'Pending' AND cost <= :free_slots
            ORDER BY priority DESC, created_at
            LIMIT 1 FOR UPDATE SKIP LOCKED)
RETURNING *;
```

```sql
-- sqlite: BEGIN IMMEDIATE takes the write lock for the transaction, so the
-- same statement without FOR UPDATE is already serialized. SKIP LOCKED has
-- no meaning and no equivalent; it does not need one.
```

That difference is the whole of the dialect split, plus `jsonb` vs `text` for the config
blobs and `now()` vs `strftime('%Y-%m-%dT%H:%M:%f','now')`. Writing it down here because a
store abstraction that hides this is one where the SQLite path is subtly not atomic and
nobody finds out until two workers exist.

### VI.2 Two resume layers, and they compose

| Layer | Owner | Granularity | What it saves |
| :--- | :--- | :--- | :--- |
| bench re-queue | scheduler | one task | the fourteen steps after the one that died |
| `pipeline-state.json`, per-script progress files | the scripts | inside a step | work the step had already finished |

They do not conflict because they answer different questions. The bench decides *whether the
step runs again*; the script decides *what it still has to do*. A `prune` that dies at 1900
of 2000 still restarts that price from zero — the granularity is a whole step, as the
sweep's header already says — and that is 1h56m at risk instead of 15h.

---

## VII. API surface

```
POST   /v1/images                    push a composed image      -> {digest, tags}
GET    /v1/images                    list
GET    /v1/images/{ref}              by digest or tag

POST   /v1/experiments               submit {image_ref, overrides} -> expands to tasks
GET    /v1/experiments               list  (the `ps` backing)
GET    /v1/experiments/{id}
DELETE /v1/experiments/{id}          cancel: pending tasks dropped, running one signalled

GET    /v1/tasks                     ?experiment= &state=
GET    /v1/tasks/{id}
GET    /v1/tasks/{id}/logs           ?follow=1  (chunked)
POST   /v1/tasks/{id}/retry          Failed | Lost -> Pending

POST   /v1/workers/register          -> worker id, slots
POST   /v1/workers/{id}/claim        lease the next admissible task
POST   /v1/tasks/{id}/heartbeat      renew lease; carries log chunk + progress
POST   /v1/tasks/{id}/complete       terminal report: exit code, artifacts, journal path
GET    /v1/workers                   who is alive, what they hold

GET    /healthz
```

FastAPI, matching `src/serve/app.py` — same framework, same `serve` optional extra, and the
same decision that `/healthz` does not touch anything slow because it is the probe.

**Auth:** none, matching `mi-lab-serve`. LAN ingress plus the cloudflared tunnel, same as
the circuit server. Worth revisiting before the API is reachable publicly, since `POST
/v1/experiments` schedules GPU work; noting it rather than solving it here.

### VII.1 Logs

The worker writes stdout and stderr to a file under the results root, as `run_step` does
today, **and** ships tail chunks with each heartbeat. The file is the archive; the store
holds a capped ring (last ~256 KB per task) so `bench logs -f` works from a laptop with no
mount. A fifteen-hour sweep's full output does not belong in Postgres.

---

## VIII. CLI

A new Typer group in `src/cli/commands/bench.py`, registered in `cli/main.py` beside the
others. Docker's verbs, because the user asked for docker's verbs and they are the right
ones:

```bash
mi-lab bench build pipelines/run/sheaf-ioi-1.7b-sweep.yaml -t ioi-sweep:v1
mi-lab bench images
mi-lab bench run ioi-sweep:v1                       # submit
mi-lab bench run ioi-sweep:v1 --set run.steps=500   # submit with overrides -> new digest
mi-lab bench ps                                     # running
mi-lab bench ps -a                                  # everything
mi-lab bench logs -f task-9c21
mi-lab bench inspect exp-7f3a
mi-lab bench cancel exp-7f3a
mi-lab bench retry task-9c21
mi-lab bench workers
```

```
$ mi-lab bench ps
EXPERIMENT  IMAGE            TASK           STATE      WORKER   ELAPSED   PROGRESS
exp-7f3a    ioi-sweep:v1     prune-s0.1     Running    gb10-0   21m       step 340/2000
exp-7f3a    ioi-sweep:v1     infer-s0.1     Blocked    -        -         needs prune-s0.1
exp-7f3a    ioi-sweep:v1     +12 more       Pending    -        -         -
exp-2b04    translation:v3   prune          Pending    -        -         -
```

`--set` composes a *new* image and prints its digest, rather than mutating the one named.
An override that changes a number changes the identity; that is invariant 4 applied to the
queue.

---

## IX. Deployment

`~/main/m/projects/k8s/mi-lab-bench/`, following the conventions of the existing
`mi-lab` directory beside it — locally built image, `IfNotPresent`, imported into the k3s
containerd, LAN ingress plus optional tunnel route.

**One image, three commands.** The same argument the circuit server's README already makes
for its two stacks: the code is identical, only the entrypoint differs. The existing
`Dockerfile` gains `scripts/bench_*.py` and nothing else.

| Deployment | Replicas | GPU | Mounts | Command |
| :--- | :--- | :--- | :--- | :--- |
| `mi-lab-bench-api` | 1 | no | — | `python -m scripts.bench_api` |
| `mi-lab-bench-scheduler` | 1, `Recreate` | no | — | `python -m scripts.bench_scheduler` |
| `mi-lab-bench-worker` | 1 | **yes** | HF cache (ro), results (**rw**) | `python -m scripts.bench_worker --slots 1` |

Three things this inherits from the circuit server's hard-won notes:

- **The worker is pinned to `gx10-d08b`.** The hostPath mounts only exist there; an unpinned
  hostPath pod lands on `mbp-m1` and `FailedMount`s.
- **The results mount is read-write**, where `mi-lab-serve` mounts the same directory
  read-only. The worker is the thing that writes circuits; the server is the thing that
  reads them. They can share the directory — publishing a circuit stays "write a folder
  under the mount", which is now something the bench does automatically.
- **`replicas: 1` with `Recreate` is the scheduler's leader election.** A second scheduler
  would double-admit. This is a deliberate non-solution: real leader election is a lease in
  the same store, and it is a later change, but two replicas must not be reachable by
  scaling this Deployment up without it.

Namespace `ai-llm`, next to the servers that share its image and its node.

---

## X. Tests

`tests/bench.py` and `tests/bench_store.py`, unittest, named after the module as the repo
requires, and **added to the offline subset in `CLAUDE.md`** — none of this needs a
checkpoint, which is the point of keeping torch out of the package.

The ones that are worth writing because they are how this breaks:

1. **Two claimers, one task.** Concurrent claims against SQLite and (if reachable) Postgres
   hand the task to exactly one worker. The test that fails if `BEGIN IMMEDIATE` is dropped.
2. **A lapsed lease is reclaimed exactly once**, and the reclaimed task returns to `Pending`
   with its epoch bumped.
3. **A fenced heartbeat is refused.** A worker presenting a stale `lease_epoch` gets a
   refusal it can act on — the guard against two processes in one results directory.
4. **The digest is stable across two files that compose to the same mapping**, and changes
   when any key that reaches a script changes.
5. **A bad key fails at build.** `run.sparsty=0.03` never reaches the queue.
6. **A pipeline expands into its steps in order, chained**, and `needs` overrides the chain.
7. **`Failed` is not auto-retried; `Lost` is.**
8. **A step's command round-trips through `--spec`** — the resolved config the image stored
   is the `SheafSpec` the script builds.
9. **The store protocol has two implementations that pass the same suite.** Parameterized
   over both backends, skipping Postgres loudly when it is unreachable, the way the adapter
   tests skip on a missing checkpoint.

---

## XI. Work plan

Ordered so that each phase is independently useful and the risky part is early.

| # | Phase | Deliverable | Useful on its own? |
| :-- | :--- | :--- | :--- |
| 1 | Records + store | `model.py`, `store.py`, `stores/{sqlite,postgres}.py`, schema, the claim/lease semantics, `tests/bench_store.py` | no — foundation |
| 2 | Images | `images.py`, `expand.py`, `scripts.sheaf --spec`, digest + tags | yes — `bench build` catches bad configs before a two-hour run |
| 3 | API | `api.py`, `scripts/bench_api.py` | yes — state is queryable and pushable |
| 4 | Worker | `worker.py`, `scripts/bench_worker.py`, heartbeat, fencing, log shipping | yes — **a submitted task survives the terminal**, the actual goal |
| 5 | Scheduler | `scheduler.py`, `scripts/bench_scheduler.py`, admission, expiry, roll-up | yes — **a reboot self-heals**, the second goal |
| 6 | CLI | `src/cli/commands/bench.py` | yes — `ps`, `logs`, `run` |
| 7 | Deploy | `~/main/m/projects/k8s/mi-lab-bench/`, Dockerfile entrypoints, manifests, scripts | yes — it outlives the host session |
| 8 | Docs | `CLAUDE.md` architecture section, offline test list, `CHEATSHEET.md` | — |

Phases 1–6 run against SQLite on the host with no cluster involved, which is also the
development loop. Phase 7 is the first time Postgres or Kubernetes is required.

**The first real test is the one that is already queued:** submit `ioi-sweep:v1`, let it run
`prune-s0.3` onward, reboot the box halfway through, and watch it come back by itself.

---

## XII. Sharp edges, written down where they bite

- **Split-brain double-execution** is the worst failure available here — two processes in
  one results directory produce a mask belonging to neither run, and `results.guard` permits
  it because both are the same config. The fencing token (§III.2) is the mitigation and
  test 3 is its receipt. If any of this is cut, this is not the part to cut.
- **The API is a single writer.** Fine at this scale, and the thing to look at first if it
  ever is not.
- **`replicas: 1` is not leader election.** Scaling the scheduler up double-admits. §IX.
- **Log volume.** Capped ring in the store, file on disk as the archive. §VII.1.
- **A worker restart loses its running task** — the container takes the process tree with
  it. Bounded by step granularity, not eliminated.
- **Cancellation of a running task is a signal, not a guarantee.** The worker kills its
  subprocess; a subprocess wedged in a CUDA call may not die promptly. `cancel` reports what
  it asked for, not what it achieved.
- **Nothing here makes a crashed *box* come back.** The 03:09 reboot is still a reboot; the
  bench only guarantees that when it does come back, the work restarts without a human. The
  underlying NVRM out-of-memory is a separate investigation and §IV is the part of it this
  proposal owns.

---

## XIII. Open question

**The package name.** `src/bench/` is used throughout this document — one word, plain, says
where experiments get run, unclaimed in this repo. `src/control/` collides in prose with
`share/schema/control.py` (a steering control); `src/queue/` names one part of it and
shadows a stdlib module by sight if not by import. If `bench` reads wrong, it is one rename
before phase 1 and free; after phase 3 it is a rename across manifests too.
