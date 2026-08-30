# Cheatsheet

Commands that have been run, with what they actually print. Everything here works on
`gpt2-small` on a CPU laptop; moving to a bigger model is the first argument changing.

## Looking at activations from a prompts file

### Check the data before any model loads

```bash
uv run python -m src.cli data check data/sentiment.prompts
```

```
sentiment-handwritten: 24 examples, 12 positive / 12 negative
balance: 50% positive
length: 32-58 characters, median 48
split at 0.3: train 16 (50%) / test 8 (50%)
no problems found
```

### Norm by layer — the cheapest check that the hook is where you think

```bash
uv run python -m src.cli viz act norms gpt2-small --data data/sentiment.prompts \
    --output charts/norms.png
```

```
dataset: sentiment-handwritten  n=24  positives=12 (50%)
captured (24, 9, 768) at layers [0, 2, 3, 4, 6, 8, 9, 10, 11]  (1.045s for 24 items (43.55 ms each))
Wrote charts/norms.png
```

The capture is `[prompts, layers, d_model]`. The chart takes the L2 norm over the last axis,
plots the mean across prompts as the line and the 10th-90th percentile as the band. **Direction
is thrown away**: two prompts with opposite meanings and the same magnitude sit on the same
point, and the labels in the file are not used at all. This is a sanity check, not a finding.

Read it for pathology: a layer at zero captured nothing, and a flat line means every layer
captured the same thing — both mean the hook is not where you think it is. Healthy is norms
climbing with depth, because each block adds to the residual stream rather than replacing it.
On gpt2-small that is ~55 at layer 0 rising to ~430 at layer 11, with a knee in the last two.

Note the defaults: nine evenly spaced depth fractions, which on a 12-layer model resolve to the
nine distinct indices above, and `position=last`. `--frac` overrides the depths, and `--show`
renders the chart inline in the terminal instead of only writing the file.

### Class separation — the same capture, with the labels used

`act norms` says nothing about whether the two classes differ. These do:

```bash
# top two principal components at every layer, coloured by label
uv run python -m src.cli viz act pca gpt2-small --data data/sentiment.prompts \
    --output charts/act-pca.png

# fit a probe at each depth; --method twice puts both on one chart
uv run python -m src.cli viz probe sweep gpt2-small --data data/sentiment.prompts \
    --method logistic --method difference_of_means --output charts/probe-sweep.png
```

```
dataset: sentiment-handwritten  n=24  positives=12 (50%)
logistic             best layer 0 at depth 0.00, AUC 0.812
difference_of_means  best layer 3 at depth 0.25, AUC 0.688
```

Do not read a number off this. `sentiment.prompts` holds 24 hand-written examples, so the test
split is 8 and an AUC moves by 0.0625 per example that changes side. Across seeds 0-4 the best
layer lands at 0, 3, 8, 4 and 2, and the best AUC between 0.50 and 0.88 -- the seed decides the
answer, which is what "too few examples" looks like from the outside. A sweep worth believing
wants hundreds of examples; `--data` with the synthetic set (`n=200`) climbs 0.95 to 1.00 with
depth, which is the shape a real sweep has.

**If a sweep comes back consistently *below* 0.5 at every layer and for both methods, that is not
noise.** Noise scatters both sides of chance. Systematically below means the probe is confidently
wrong, and the usual cause is a contrast pair straddling the split: the probe learns the subject
from the training half, and every test twin is that same subject carrying the opposite label. Check
`load_labeled(path).groups` -- if it is `None`, the file never declared its pairs, and indenting
the second line of each pair fixes it.

### The rest of the `viz act` group

| command | takes | shows |
| --- | --- | --- |
| `heatmap` | one `-p` prompt | layers by dimensions; a vertical stripe is an outlier dimension |
| `tokens` | one `-p` prompt | norm at every (layer, token position), so `position=all` |
| `similarity` | `--data` or `-p` | cosine between every pair of layers, over all layers |
| `drift` | `--data` plus `--against <config>` | two models layer-for-layer, cosine and relative L2 |

## Importing a downloaded dataset

Downloads arrive as CSV, and `--data` reads only `.prompts`/`.jsonl` — so a CSV is converted once
and kept, rather than being a format the pipeline reads at run time.

```bash
# fetch (geometry-of-truth: 748 true/false pairs over cities, and the same statements negated)
mkdir -p data/external/geometry-of-truth
for f in cities.csv neg_cities.csv; do
  curl -sSfL -o "data/external/geometry-of-truth/$f" \
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/$f"
done

# convert, naming the column that identifies a pair
uv run python -m src.cli data convert data/external/geometry-of-truth/cities.csv \
    --out data/cities.prompts --text-field statement --group-field city \
    --labels "false,true" --name cities

uv run python -m src.cli data convert data/external/geometry-of-truth/neg_cities.csv \
    --out data/neg-cities.prompts --text-field statement --group-field city \
    --labels "false,true" --name neg-cities
```

```
wrote 1496 examples (748 groups) to data/cities.prompts
```

**`--group-field` is the whole point of this command.** Without it the 1496 rows import as 1496
independent examples, the pairs straddle the split, and the sweep comes back below chance. With it,
rows sharing a `city` become one group and `dumps` writes the pair as an indented run:

```
+ The city of Krasnodar is in Russia.
  - The city of Krasnodar is in South Africa.
```

A group's rows must be adjacent in the CSV. Scattered rows are refused by line number --
`cities.csv:4 belongs to group 'Krasnodar', which stopped earlier in the file` -- because a group
written non-adjacently could not be read back the same way, and a converted file that quietly loses
its pairing is the failure the whole argument exists to prevent.

Always `data check` the result before training on it:

```
cities: 1496 examples, 748 true / 748 false
groups: 748 kept whole by a split, sizes [2]
split at 0.3: train 1048 (50%) / test 448 (50%)
no problems found
```

The `.prompts` file records no provenance of its own -- `dumps` writes `name:` and `labels:` and
drops prose -- so the conversion command above is the provenance. Note also that
`saprmarks/geometry-of-truth` ships no LICENSE file, which by default means all rights reserved:
fine locally, not something to redistribute.

### What the sweep looks like on 1496 examples

```bash
uv run python -m src.cli viz probe sweep gpt2-small --data data/cities.prompts \
    --method logistic --method difference_of_means --output charts/probe-sweep-cities.png
```

```
logistic             best layer 11 at depth 0.92, AUC 0.871
difference_of_means  best layer 11 at depth 0.92, AUC 0.771
```

Logistic climbs 0.661 to 0.871 and plateaus after depth 0.33; difference-of-means tracks it about
0.1 lower. Seeds 0, 1 and 2 all pick layer 11 and give 0.871 / 0.879 / 0.901 -- the same winner and
the same curve three times, which is what the 24-example set could not do. With 448 test examples
one example moves the AUC by 0.002, so the trend is now larger than the noise.

Two things not to over-read. "Layer 11 wins" is the weakest claim on the chart: from depth 0.33 the
curve is flat within 0.03, and the plateau is the finding rather than the argmax. And the probe may
be reading plausibility rather than truth -- "Krasnodar is in South Africa" is both false and
statistically odd. `likely.csv` in the same repo is the control for exactly that.

### Transfer: train on the statements, test on their negations

`sweep(adapter, train, test)` takes any two datasets, so this needs no new code -- but it does need
care about subjects. `cities` and `neg-cities` share all 748 cities, so training on the whole of one
and testing on the whole of the other lets a probe that memorized the subject look like a probe that
transferred. The two files are row-aligned (row `i` is the same city and country, negated, label
flipped), so the same 224 held-out cities can be taken from both sides.

```
1048 train rows / 448 test rows, 524 train cities / 224 held-out cities, no city on both sides

cities -> cities           best layer 11  auc 0.871   all: [0.66, 0.77, 0.8, 0.84, 0.85, 0.86, 0.85, 0.85, 0.87]
cities -> neg-cities       best layer  0  auc 0.343   all: [0.34, 0.26, 0.23, 0.17, 0.22, 0.22, 0.2, 0.23, 0.18]
neg-cities -> neg-cities   best layer  9  auc 0.863   all: [0.69, 0.78, 0.78, 0.84, 0.84, 0.86, 0.86, 0.85, 0.86]
neg-cities -> cities       best layer  0  auc 0.318   all: [0.32, 0.23, 0.24, 0.16, 0.15, 0.18, 0.14, 0.14, 0.14]
```

**The probe does not transfer, and it fails in the informative direction.** Each probe scores ~0.87
on its own kind of statement and ~0.15 on the negated kind -- not near chance, but close to the
exact inverse (`1 - 0.871 = 0.129`). It is symmetric: training on the negations and testing on the
affirmatives fails the same way.

The direction says the same thing:

```
cosine(cities probe, neg-cities probe): -0.983 at layer 0, rising to -0.72 at layer 8, -0.865 at 11
```

Nearly antiparallel at every depth. So the two probes found *the same feature* and disagree only
about which label it means. That feature cannot be "true" -- it behaves like the association between
the city and the country, which "Lodz is not in Poland" leaves intact while flipping the label. The
0.871 in the sweep above is a probe reading plausibility, exactly the caveat that section flagged.

This is the reason `neg_cities.csv` exists in that repo, and the reason to be suspicious of a
single-dataset probe: a probe that scores 0.87 in-distribution and inverts under a one-word change
is not measuring what its dataset name says it measures.


## Replicating the IOI circuit

Wang et al. (2023) reverse-engineered the circuit GPT-2 small uses to finish *"Then, John and Mary
went to the store. Mary gave a ring to \_\_"* with **John**. `src.cli ioi` replays that argument end
to end on a CPU laptop.

### Build the prompts and see what is being asked

```bash
uv run python -m src.cli ioi dataset gpt2-small --size 4 --show 2
```

```
ioi-abc: 4 pairs, ABBA share 50%
frame: Then, {first} and {second} went to the {place}. {third} gave a {object} to

[ABBA] Then, Jack and Sam went to the park. Sam gave a ball to
           Then, Jack and Sam went to the park. Mary gave a ball to
           -> Jack, not Sam

[BABA] Then, Jack and Lee went to the station. Jack gave a drink to
           Then, Jack and Lee went to the station. Bob gave a drink to
           -> Lee, not Jack

positions:
    0  'Then'
    ...
    2  ' Jack'  <- IO
    4  ' Sam'   <- S1
   10  ' Sam'   <- S2
   14  ' to'    <- END
```

Three things are enforced rather than hoped for, and all three are silent failures otherwise:
**one frame per dataset** and **single-token names**, so every prompt lines up position for
position and patching compares like with like; and **the two name orders alternated**, so a model
that always answers the first name scores half rather than everything.

`--corruption swap` exchanges the two roles instead of replacing the repeated name. It opens a
wider span (the correct answer flips rather than merely blurring) at the cost of changing two
things at once.

### Does the model do the task at all

```bash
uv run python -m src.cli ioi evaluate gpt2-small --size 32
```

```
32 prompts  accuracy 100.0%  logit diff +2.965 clean / -0.049 corrupted (span +3.014)
```

**The span is the number to read.** Every later result is a fraction of it, so a corruption that
did not corrupt anything turns the rest of the group into confident noise — `evaluate` warns when
the span drops under 0.5.

### What each head writes: direct logit attribution

```bash
uv run python -m src.cli ioi attribute gpt2-small --size 8 --top 4
```

```
measured logit difference : +2.959
attributed to components  : +2.959
unattributed remainder    : +1.72e-06
  embedding +0.014   biases and final shift +0.142

towards the answer (top 4):        against the answer (top 4):
  L9  H9   +2.625                    L10 H7   -1.740
  L9  H6   +0.976                    L11 H10  -1.080
  L10 H2   +0.472                    L11 H1   -0.280
  L10 H10  +0.400                    L11 H3   -0.116
```

One forward pass answers for every component, because the residual stream is a sum. **Check the
remainder first**: it is the receipt on the decomposition, and 1.7e-06 means every site the model
writes into is one the adapter hooks. L9H9 and L9H6 are the paper's Name Movers; L10H7 and L11H10
are its Negative Name Movers.

### Where each head looks

```bash
uv run python -m src.cli ioi heads gpt2-small --size 8
```

```
      name mover: L9H6, L9H9, L10H7, L11H10
    s-inhibition: L7H9, L8H5, L8H6, L8H10
 duplicate token: L1H11, L3H0
       induction: L5H5
```

Each role is a query position *and* a key position — a head's job is where it looks from as much as
where it looks to. Attention names candidates; only patching promotes one to a finding.

### Where the answer gets decided

```bash
uv run python -m src.cli ioi patch gpt2-small --size 8 --site residual
```

```
span to recover: +2.553 logits
 layer      0      1     IO      3     S1      5      6      7      8      9     S2     11     12     13    END
     0   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.01  -0.00  -0.00  -0.00  -0.00
     ...
     6   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.79   0.02   0.00   0.00   0.04
     7   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.46   0.01   0.00   0.00   0.38
     8   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.05   0.01  -0.00   0.00   0.99
     ...
    11   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   0.00   1.00
```

Read it as a story: the whole answer sits on **S2** (the repeated name) through layer 6, and hands
off to **END** at layers 7-8. That handoff is the moment the answer stops being a fact about a
token earlier in the sentence and becomes a fact about what comes next. Costs one forward pass per
cell — twelve layers by fifteen positions is about 40s on a laptop.

### Growing a circuit and checking it

```bash
uv run python -m src.cli ioi circuit gpt2-small --size 8
```

```
7 heads: L10H7, L11H10, L8H6, L7H9, L8H10, L5H5, L9H9
   1. + L10H7   recovery -0.524
   2. + L11H10  recovery -1.128
   ...
   7. + L9H9    recovery +0.919

faithfulness (this set alone restores the answer) : 0.919
necessity    (the clean run needs it)             : 1.160
minimality   (recovery lost by dropping each head) :
  L10 H7   -0.065  spare  [name mover]
  L11 H10  -0.069  spare  [name mover]
  L8  H6   +0.318  load-bearing  [s-inhibition]
  L7  H9   +0.320  load-bearing  [s-inhibition]
  L8  H10  +0.214  load-bearing  [s-inhibition]
  L5  H5   +0.134  load-bearing  [induction]
  L9  H9   +0.192  load-bearing  [name mover]

2 head(s) the circuit does not need: L10H7, L11H10
```

Three checks together, because any one alone misleads. A faithful circuit that is not necessary is
one route among several; a necessary one full of passengers claims more than the evidence supports.

The greedy search ranks candidates by *absolute* effect, so it opens with the two Negative Name
Movers and the recovery curve **goes down before it goes up** — restoring a head that pushes against
the answer makes things worse. Minimality then reports both as spare with a negative cost: dropping
them from the circuit raises faithfulness.

### The whole battery on one page

```bash
uv run python -m src.cli viz circuit dashboard gpt2-small --size 8 --out-dir charts/ioi
```

Eight charts and a self-contained `circuit.html`, all measured off one dataset and one pair of
baselines — which is what stops being true the moment they are made one command at a time. The
chart worth looking at is `circuit-compare.png`: direct attribution on one axis, causal patching on
the other. **L9H9 writes 2.6 logits and patches to 0.22; L7H9 and L8H6 write nothing and patch to
0.28.** The heads off the diagonal are the ones neither method finds alone, and the disagreement is
the result rather than a bug.

### As a run

```bash
uv run python -m src.cli run exec --preset ioi-circuit --set ioi.size=16
```

```
completed in 231.1s -> outputs/ioi-circuit/20260826-035015-82807f2efe6d
  accuracy: 1
  attribution_remainder: 5.10458e-06
  best_patch_layer: 0
  best_patch_position: 10
  best_patch_recovery: 1.00898
  clean_logit_difference: 2.96518
  corrupted_logit_difference: -0.0492563
  faithfulness: 0.903974
  necessity: 1.08049
  n_heads: 7
  n_prompts: 16
  n_spare: 2
  span: 3.01444
  produced artifact: circuit.mia
```

The grids that are really matrices — per-head attribution, per-head effect, the role weights, the
position map — go into `circuit.mia` beside `run.json`, because a metrics dict with one entry per
head is a file nobody reads. `run replay <dir>` re-runs it from the spec it recorded.

Runs are filed under the experiment that produced them, `outputs/<experiment>/<run id>/`, because a
root accumulates runs forever and the question asked of it is nearly always "what did *this*
experiment do". Point the chart battery at the same directory and the run becomes self-contained:

```bash
RUN=outputs/ioi-circuit/20260826-035015-82807f2efe6d
uv run python -m src.cli viz circuit dashboard gpt2-small --size 16 --out-dir "$RUN/charts"
```

```
$RUN/
  spec.yaml        the resolved spec, reproducing hash 82807f2efe6d
  run.json         status, metrics, produced refs, duration
  circuit.mia/     artifact.json + tensors.safetensors
  charts/          eight PNGs and a self-contained circuit.html
```

The dashboard rebuilds the dataset from the same seed, frame and corruption, so its panels report
the same 7 heads, faithfulness 0.90 and necessity 1.08 that `run.json` does. `outputs/` is
gitignored — the run is reproducible from its own `spec.yaml`, which is the thing worth keeping.

### Sharing the result

`run.json` is the record of what this machine did; `circuit.mia` is the part meant to leave it. It
is a directory holding a JSON card and one `safetensors` file, and reading it costs no checkpoint:

```bash
uv run python -m src.cli artifact show outputs/ioi-circuit/20260826-035015-82807f2efe6d/circuit.mia
uv run python -m src.cli artifact check outputs/ioi-circuit/20260826-035015-82807f2efe6d/circuit.mia
```

```
circuit  ioi-abc-gpt2-small  (0.1, 2026-08-26T02:47:10+00:00)
model    : gpt2-small (gpt2), 12 layers x 12 heads, float32
site     : head_out at layers [0, 1, 2, 3, 4, 5] ... (depth 0.00, 0.08, 0.17, ...), position all
baseline : logit_difference +2.959 clean / +0.406 corrupted (span +2.553)
  faithfulness           +0.9188
  necessity              +1.1603
tensors  :
  head_attribution       layer=12 x head=12  [logits]
  head_effects           layer=12 x head=12  [recovery]
  residual_patch         layer=12 x position=15  [recovery]
circuit  : 7 nodes, 0 measured edges
  L10H7    causal -0.524  attribution -1.740  name mover
  L9H9     causal +0.226  attribution +2.625  name mover
made by  : mi-lab at 0c248f7, torch 2.13.0+cu130
```

Every tensor carries its axes and their tick labels, so the position map is redrawable — and
differenceable against another model's — by a tool that never ran GPT-2. Every node carries what
*both* halves of the study said about it, which is the only way the L9H9 / L10H7 disagreement above
survives being shared. The same envelope wraps a probe:

```bash
uv run python -m src.cli artifact pack probe.pt
uv run python -m src.cli probe score gpt2-small --probe probe.mia -p "I adored the concert"
```

`probe score` and `steer probe` take either the `.pt` or the `.mia`, and give identical numbers.
`docs/artifact-format.md` is the format itself.

## Comparing the ways of finding a circuit

The `ioi` group answers "which heads do this task". This group answers the three questions that come
after it: do the techniques agree about which heads, is it the same circuit on every example, and
would that circuit have cost another task any less. Nothing here needs a model until you ask it to:

```bash
uv run python -m src.cli compare list
```

```
tasks:
  agreement      which verb form the subject takes, across an attractor
  greater_than   a year later than the one just named (Hanna et al., 2023)
  induction      copy what followed this token last time (Olsson et al., 2022)
  ioi            which of two names the sentence has not used yet (Wang et al., 2023)

techniques:
  ablation       how much of the clean behaviour is lost when each head alone is knocked out
                 in recovery, one forward pass per head, plus three for the baselines
  attribution    what each head wrote towards the answer along the direct path to the unembedding
                 in logits, two forward passes, whatever the model's size
  eap            a gradient estimate of what patching each head would have recovered
                 in recovery (first order), one backward pass and four forward passes
  patching       how much of the corruption's span each head restores when written back on its own
                 in recovery, one forward pass per head, plus three for the baselines
  random         a seeded shuffle, so a circuit's numbers can be read against a circuit of the same size
```

### Does the model do the other tasks at all

A circuit for a task the model fails is a circuit for failing, and the span is what every later
number is a fraction of. Check both before anything expensive:

```bash
uv run python -m src.cli compare tasks gpt2-small --size 16
```

```
task             n  accuracy    clean  corrupted    span
agreement       16     100%    +2.61      -3.86   +6.47
greater_than    16     100%    +5.46      -1.69   +7.15
induction       16     100%    +4.21      -0.18   +4.40
ioi             16     100%    +2.96      +0.41   +2.55
```

### Five techniques, one task, one circuit size

Same size for all of them, because faithfulness climbs with the head count and a comparison at
different sizes is a comparison of sizes:

```bash
uv run python -m src.cli compare techniques gpt2-small --task ioi --size 16 --count 8
```

```
technique      faithful  necessity  incomplete  passes  seconds
ablation         +0.966     +1.237       0.237     147     25.5
attribution      +0.526     +0.459       1.426       2      0.4
eap              +1.007     +1.270       0.270       5      0.9
patching         +1.007     +1.270       0.270     147     30.9
random           -0.193     -0.047       1.195       0      0.0

  ablation     L10H7, L8H10, L8H6, L5H5, L11H10, L7H9, L9H9, L3H0
  attribution  L9H9, L10H7, L9H6, L11H10, L10H1, L10H10, L8H10, L10H6
  eap          L10H7, L8H10, L11H10, L8H6, L7H9, L9H9, L5H5, L7H3
  patching     L10H7, L8H10, L11H10, L8H6, L5H5, L7H9, L9H9, L7H3
  random       L4H3, L5H9, L11H10, L11H8, L8H4, L9H5, L2H2, L6H9

circuit overlap (intersection over union):
                  ablation  attribution          eap     patching       random
ablation              1.00         0.33         0.78         0.78         0.07
attribution           0.33         1.00         0.33         0.33         0.07
eap                   0.78         0.33         1.00         1.00         0.07
patching              0.78         0.33         1.00         1.00         0.07
random                0.07         0.07         0.07         0.07         1.00

rank correlation:
                  ablation  attribution          eap     patching       random
attribution           0.20         1.00         0.21         0.22         0.03
eap                   0.73         0.21         1.00         1.00         0.11
```

The `seconds` column is CPU wall-clock; `passes` is not device-dependent, which is why the argument
below is made in passes. With `device: auto` these configs land on a GPU where there is one and every
second in that column shrinks, while the column beside it does not move.

Three things in that table. **eap selected patching's eight heads exactly**, for five forward passes
against a hundred and forty-seven — that is the whole argument for the gradient approximation, and
it is a measurement here rather than a citation. **Direct attribution agrees at 0.22**, and its
circuit recovers half of what patching's does: the direct path is a different question, not a cheap
version of the same one. **The random control lands at -0.19**, which is what the other numbers have
to be read against.

`ablation` sits between them at 0.78 overlap, and it is the one worth staring at: it asks what
removing a head costs rather than what restoring it recovers, and the two only come apart where the
model has a second route to the answer.

### Is it the same circuit twice

A circuit found on a batch is an average, and an average can be made of components no single example
used. So run the technique once per example — which is why the default is the one whose cost does
not scale with the model:

```bash
uv run python -m src.cli compare consistency gpt2-small --task ioi --size 16 --count 8
```

```
eap on ioi-abc: 16 per-example circuits of 8, 8 shared at P=0.5, reuse 0.81 (chance 0.03)

how often each head was selected:
  L8  H10   100%  shared
  L7  H9     94%  shared
  L10 H7     94%  shared
  L11 H10    88%  shared
  L8  H6     81%  shared
  L5  H5     75%  shared
  L9  H9     69%  shared
  L5  H9     50%  shared
  L9  H6     25%
  L3  H0     19%
```

Reuse 0.81 against a chance line of 0.03: on this task the batch-level circuit is a real description
of what happens example by example, and the tail below 25% is where the averaging was hiding
something.

### Is it about the task

The check the diagonal cannot make. Find a circuit on every task with the same technique at the same
size, then ablate each on all of them:

```bash
uv run python -m src.cli compare specificity gpt2-small --size 16 --count 8
```

```
  agreement     L7H4, L10H9, L8H5, L10H5, L11H10, L5H2, L6H0, L6H5
  greater_than  L9H1, L7H10, L8H11, L5H5, L5H1, L6H9, L8H8, L10H7
  induction     L6H9, L10H7, L9H6, L7H10, L5H5, L9H9, L11H10, L10H1
  ioi           L10H7, L8H10, L11H10, L8H6, L7H9, L9H9, L5H5, L7H3

damage (row = measured on, column = circuit ablated), as a share of the clean logit difference:
                   agreement  greater_than     induction           ioi        random
agreement               0.90         -0.00          0.09          0.07         -0.04
greater_than            0.01          0.18          0.01          0.01          0.01
induction              -0.02          0.27          0.54          0.01         -0.02
ioi                    -0.24          0.30          0.38          1.18          0.07

  ioi            own +1.18  others +0.15  margin +1.04
  agreement      own +0.90  others +0.05  margin +0.85
  induction      own +0.54  others +0.09  margin +0.45
  greater_than   own +0.18  others +0.01  margin +0.17

circuit overlap between tasks (two independent picks of this size share 0.03):
  greater_than   vs induction      0.33
  induction      vs ioi            0.33
  agreement      vs ioi            0.07
```

The greater-than circuit is L5H5, L6H9, L7H10, L8H11, L9H1 and three others — the heads Hanna et al.
name, found here by a gradient and eight forward passes.

Two readings that are easy to get backwards. The margins are large, so **at this size and at this
granularity these circuits do look specific** — which is not what the cross-task literature reports,
and the difference is the setup rather than the result: eight attention heads is 6% of this model's
144, the recent comparisons select 10–30% of components *including MLP layers*, and it is the shared
MLPs they attribute the overlap to. Nothing here selects an MLP, so this measures the specificity of
the attention circuit and says nothing about the rest.

And `greater_than` losing only 0.18 to its own circuit is the second reading: mean ablation removes
only what *varied across the batch*, and every task here is one frame with one slot changing. A
near-zero diagonal is that as often as it is a task the attention heads do not carry — Hanna et al.
found greater-than runs largely through MLPs, so here it is probably both.

### The whole thing as a run

```bash
uv run python -m src.cli run exec --preset circuit-comparison
```

The preset runs all five techniques on IOI, consistency at P=0.5, and the cross-task sweep over all
four tasks, then writes `comparison.mia` beside `run.json`. That artifact is the first one this repo
produces with its **control slots full**: `random_baseline` carries the same-size random circuit's
faithfulness and `cross_task` carries one entry per other task. Every other artifact here ships them
empty and means it — nobody checked — and `artifact check` says so:

```bash
uv run python -m src.cli artifact check outputs/circuit-comparison/*/comparison.mia
```

The charts follow the same shape, all six measured off one comparison so the panels are about each
other:

```bash
uv run python -m src.cli viz compare dashboard gpt2-small --size 16 --count 8
```

Six panels land in `charts/compare/`, and `cost.png` is the one to look at first: forward passes
across on a log scale, agreement with patching up. A technique in the top left is one to run on a
model where patching cannot be run at all.
