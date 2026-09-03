"""Guards on the pruner, most of which exist because a number looked too good.

A weight mask has one gate per parameter -- 85M of them on GPT-2 small -- and
whatever it is scored on, it can memorize. The first run of this reported 0.02%
of weights open at accuracy 1.000, which was twenty thousand weights having
learned eight sentences. These tests are what stops that being reported again.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
import torch.nn.functional as functional

from src.data.tasks import build_task
from src.methods.circuits import CircuitError, require_circuits
from src.methods.sheaves import (
    _chunk,
    _split,
    gateable,
    gumbel_sigmoid,
    load_bearing,
    prune,
    schedule,
    span,
    target_schedule,
)
from src.telemetry.journal import Journal, read_metrics

from .stubs.model import shared_adapter


class TestGates(TestCase):
    def test_the_gate_is_hard_forward_and_differentiable_backward(self):
        """Straight-through: exactly 0 or 1 out, a usable gradient in"""
        logits = torch.zeros(2048, requires_grad=True)
        gate = gumbel_sigmoid(logits, temperature=1.0)
        self.assertTrue(bool(((gate == 0) | (gate == 1)).all()), "the gate is not hard in the forward pass")
        gate.sum().backward()
        self.assertTrue(bool((logits.grad != 0).any()), "no gradient reached the logits")

    def test_a_high_logit_opens_and_a_low_one_closes(self):
        self.assertGreater(float(gumbel_sigmoid(torch.full((4096,), 8.0)).mean()), 0.95)
        self.assertLess(float(gumbel_sigmoid(torch.full((4096,), -8.0)).mean()), 0.05)

    def test_temperature_changes_the_gradient_and_not_the_gate(self):
        """The paper's eq. 3 adds the noise before dividing by tau: which gates open is tau-free"""
        logits = torch.linspace(-3, 3, 4096, requires_grad=True)
        torch.manual_seed(0)
        cold = gumbel_sigmoid(logits, temperature=0.01)
        torch.manual_seed(0)
        warm = gumbel_sigmoid(logits, temperature=1.0)
        self.assertTrue(torch.equal(cold, warm))
        cold.sum().backward()
        sharp = logits.grad.clone()
        logits.grad = None
        warm.sum().backward()
        self.assertGreater(float(sharp.max()), 10 * float(logits.grad.max()))

    def test_no_noise_is_the_thresholded_mask(self):
        """At noise 0 the sample is `logits > 0`: what `anneal` converges the training on"""
        logits = torch.linspace(-3, 3, 4096, requires_grad=True)
        gate = gumbel_sigmoid(logits, noise=0.0)
        self.assertTrue(torch.equal((logits > 0).float(), gate))
        gate.sum().backward()
        self.assertTrue(bool((logits.grad > 0).all()), "the noiseless gate still has to be differentiable")
        torch.manual_seed(0)
        half = gumbel_sigmoid(logits.detach(), noise=0.5)
        self.assertLess(float((half != (logits > 0).float()).float().mean()),
                        float((gumbel_sigmoid(logits.detach()) != (logits > 0).float()).float().mean()))

class TestTargetSchedule(TestCase):
    def test_the_target_tightens_from_everything_and_then_holds(self):
        self.assertAlmostEqual(1.0, target_schedule(0, 0.05, 100))
        self.assertAlmostEqual(0.525, target_schedule(50, 0.05, 100))
        self.assertAlmostEqual(0.05, target_schedule(100, 0.05, 100))
        self.assertAlmostEqual(0.05, target_schedule(900, 0.05, 100))
        self.assertAlmostEqual(0.05, target_schedule(0, 0.05, 0))

class TestSchedule(TestCase):
    def test_the_price_ramps_and_then_holds(self):
        """A constant price picks one of two failures; the reference ramps 1000x"""
        self.assertAlmostEqual(1.0, schedule(0, 1.0, 1000.0, 100))
        self.assertAlmostEqual(1000.0, schedule(100, 1.0, 1000.0, 100))
        self.assertAlmostEqual(1000.0, schedule(500, 1.0, 1000.0, 100))
        self.assertLess(schedule(10, 1.0, 1000.0, 100), schedule(50, 1.0, 1000.0, 100))

class TestPruning(TestCase):
    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)

    def test_a_task_too_small_to_hold_out_is_refused(self):
        """The guard on the failure that produced this module's first result

        A mask scored on the prompts it trained on reports memorization as
        faithfulness, and at 85M gates it can memorize anything.
        """
        tiny = build_task("ioi", self.adapter, size=2, seed=0)
        with self.assertRaises(CircuitError):
            prune(self.adapter, tiny, steps=1, holdout=0.0)

    def test_zero_steps_is_refused(self):
        task = build_task("ioi", self.adapter, size=8, seed=0)
        with self.assertRaises(CircuitError):
            prune(self.adapter, task, steps=0)

    def test_the_weights_are_restored_afterwards(self):
        """Masking runs through functional_call, so the model must come back untouched"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        before = self.adapter.logits(list(task.clean)).clone()
        prune(self.adapter, task, steps=2, batch=4, holdout=0.25)
        self.assertTrue(torch.equal(before, self.adapter.logits(list(task.clean))))

    def test_it_reports_train_and_held_out_separately(self):
        """Both numbers, always: their gap is the finding on a task this size"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        sheaf = prune(self.adapter, task, steps=2, batch=4, holdout=0.25)
        self.assertIsNotNone(sheaf.train_accuracy)
        self.assertIsNotNone(sheaf.accuracy)
        self.assertGreaterEqual(sheaf.density, 0.0)
        self.assertLessEqual(sheaf.density, 1.0)

    def test_protected_weights_stay_open_whatever_the_price(self):
        """The largest weights by magnitude are pinned, and counted as open"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        sheaf = prune(self.adapter, task, steps=3, batch=4, holdout=0.25, sparsity=1e6,
                      init=-1.0, protect=0.01, probe_every=0)
        self.assertGreater(sheaf.n_pinned, 0)
        self.assertLessEqual(sheaf.n_pinned, int(0.0101 * sheaf.n_parameters) + len(sheaf.gates))
        # started shut and priced at a million: the only open gates are the pinned ones
        self.assertEqual(sheaf.n_pinned, sheaf.n_open)
        largest = max(((float(w.abs().max()), name) for name, w in
                       ((n, p) for n, p in self.adapter.model.named_parameters() if n in sheaf.gates)))
        name = largest[1]
        w = dict(self.adapter.model.named_parameters())[name].detach().abs().flatten()
        self.assertGreater(float(sheaf.gates[name].flatten()[int(w.argmax())]), 0.0)

    def test_a_target_learns_its_price_and_a_bad_one_is_refused(self):
        """The multipliers climb while the mask is denser than the target"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        with self.assertRaises(CircuitError):
            prune(self.adapter, task, steps=1, batch=4, target=1.5)
        with TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "run", params={})
            prune(self.adapter, task, steps=4, batch=4, holdout=0.25, journal=journal,
                  probe_every=0, target=0.01, warmup=1, faith_kind="nll")
            journal.finish()
            rows = read_metrics(Path(directory) / "run")
        self.assertEqual([1.0, 0.01, 0.01, 0.01], [row["target"] for row in rows])
        # The gap is measured on the hard density, which starts at exactly
        # 1.0 against a target of 1.0: nothing to pay at step 0, and the price
        # climbs from there while the mask is denser than the target.
        self.assertEqual(0.0, rows[0]["lambda1"])
        self.assertEqual(1.0, rows[0]["density"])
        self.assertGreater(rows[-1]["lambda1"], 0.0, "denser than the target, the price should rise")
        self.assertGreater(rows[-1]["price"], 0.0)
        self.assertTrue(all("density" in row for row in rows), "the constraint is journaled every step")

    def test_the_probe_scores_the_thresholded_mask_while_training(self):
        """The loss is a sampled mask's; the result is the thresholded one's. The
        curve has to carry the second, or a collapse at the threshold is
        invisible until the run is over -- which is how the 1.7B spent 1h49m
        on a mask that was at chance from its first probe."""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        with TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "run", params={})
            prune(self.adapter, task, steps=3, batch=4, holdout=0.25, journal=journal, probe_every=2,
                  anneal=True, init=5.0, temperature=0.01)
            journal.finish()
            rows = read_metrics(Path(directory) / "run")
        probed = [row for row in rows if "density" in row]
        self.assertEqual([0, 2], [row["step"] for row in probed])
        for row in probed:
            self.assertIn("hard_accuracy", row)
            self.assertGreaterEqual(row["hard_accuracy"], 0.0)
            self.assertLessEqual(row["hard_accuracy"], 1.0)
        self.assertNotIn("hard_accuracy", rows[1])

class TestBand(TestCase):
    """The layer band, which is what makes a model past GPT-2 affordable at all"""

    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)

    def test_a_band_gates_only_its_own_layers(self):
        """Two layers out of twelve, and the count has to fall with them"""
        whole = gateable(self.adapter)
        band = gateable(self.adapter, [0, 1])
        self.assertLess(sum(p.numel() for p in band.values()),
                        sum(p.numel() for p in whole.values()))
        self.assertTrue(all("h.0." in name or "h.1." in name for name in band),
                        f"a band of layers 0-1 reached other layers: {sorted(band)}")

    def test_a_layer_outside_the_model_is_refused(self):
        """The silent version aims at a layer that is not there and gates nothing"""
        with self.assertRaises(CircuitError):
            gateable(self.adapter, [0, len(self.adapter.blocks)])

    def test_an_empty_band_is_refused(self):
        with self.assertRaises(CircuitError):
            gateable(self.adapter, [])

    def test_the_band_is_carried_onto_the_sheaf(self):
        """`density` is a fraction of what was gated, so the sheaf has to say what that was"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        sheaf = prune(self.adapter, task, steps=2, batch=4, holdout=0.25, layers=[0, 1])
        self.assertEqual([0, 1], sheaf.layers)
        self.assertIn("layers 0-1", str(sheaf))

    def test_gates_are_float32_whatever_the_weights_are(self):
        """bfloat16 gates underflow AdamW's second moment and cannot sum over a billion terms"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        sheaf = prune(self.adapter, task, steps=1, batch=4, holdout=0.25, layers=[0])
        self.assertTrue(all(logits.dtype is torch.float32 for logits in sheaf.gates.values()))

class TestSpan(TestCase):
    def test_a_contiguous_band_reads_as_a_range(self):
        self.assertEqual("21-27", span([21, 22, 23, 24, 25, 26, 27]))
        self.assertEqual("21,23,26", span([26, 21, 23]))
        self.assertEqual("5", span([5]))

class TestChunking(TestCase):
    """`--size` has to reach training, which for 500 steps of this it did not"""

    def test_the_batch_walks_the_training_rows(self):
        rows = list(range(12))
        self.assertEqual([0, 1, 2, 3], _chunk(rows, 0, 4))
        self.assertEqual([4, 5, 6, 7], _chunk(rows, 1, 4))
        self.assertEqual([8, 9, 10, 11], _chunk(rows, 2, 4))
        self.assertEqual([0, 1, 2, 3], _chunk(rows, 3, 4))

    def test_every_row_is_seen_within_one_pass(self):
        rows = list(range(12))
        seen = {row for step in range(3) for row in _chunk(rows, step, 4)}
        self.assertEqual(set(rows), seen, "a full pass did not reach every training row")

    def test_a_batch_at_least_as_wide_as_the_rows_is_all_of_them(self):
        self.assertEqual([0, 1, 2], _chunk([0, 1, 2], 7, 8))
        self.assertEqual([0, 1, 2], _chunk([0, 1, 2], 7, 0))

    def test_it_wraps_rather_than_running_short(self):
        """A batch straddling the end must still be a full batch"""
        self.assertEqual([3, 4, 0], _chunk(list(range(5)), 1, 3))
        self.assertEqual(4, len(_chunk(list(range(6)), 1, 4)))

class TestLoadBearing(TestCase):
    """The check that tells a circuit apart from a band nothing needed"""

    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)

    def test_shutting_the_whole_model_costs_the_task(self):
        """Gating everything and closing it leaves no network, so the score must fall"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        control = load_bearing(self.adapter, task)
        self.assertGreater(control["open"] - control["shut"], 0.0,
                           "shutting every weight in the model did not cost any accuracy, "
                           "which means the mask is not reaching the forward pass")

    def test_open_matches_the_unmasked_model(self):
        """An all-open band is the model itself; if it is not, masking is broken"""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        # `logits` is already [batch, vocab] at each prompt's final real token
        logits = self.adapter.logits(list(task.clean))
        io, subject = task.answers(self.adapter)
        unmasked = float(sum(
            1.0 for row in range(logits.shape[0])
            if logits[row, io[row]] > logits[row, subject[row]]
        ) / logits.shape[0])
        control = load_bearing(self.adapter, task, layers=[0, 1])
        self.assertAlmostEqual(unmasked, control["open"], places=6,
                               msg="an all-open band did not reproduce the unmasked model")
        self.assertGreaterEqual(control["open"], control["shut"])

    def test_it_scores_only_the_rows_it_is_given(self):
        task = build_task("ioi", self.adapter, size=8, seed=0)
        control = load_bearing(self.adapter, task, layers=[0], rows=[0, 1])
        self.assertIn(control["open"], (0.0, 0.5, 1.0), "two rows cannot score anything else")

class TestSplit(TestCase):
    """The holdout, defeated twice already: once by scoring train on train, once
    by a task whose pool is smaller than the split it was cut with."""

    def test_no_prompt_lands_on_both_sides(self):
        prompts = ["a", "b", "c", "a", "b", "d", "a", "e"]
        train, test = _split(prompts, 0.25)
        self.assertEqual(set(), {prompts[row] for row in train} & {prompts[row] for row in test},
                         "a prompt appears in both the training and the held-out rows")

    def test_every_row_is_placed_exactly_once(self):
        prompts = ["a", "b", "c", "a", "b", "d", "a", "e"]
        train, test = _split(prompts, 0.25)
        self.assertEqual(list(range(len(prompts))), sorted(train + test))

    def test_rows_repeating_one_prompt_leave_no_holdout(self):
        """128 rows of 1 prompt is 1 prompt, and the old split called it 96/32"""
        with self.assertRaises(CircuitError):
            _split(["same"] * 128, 0.25)

    def test_the_holdout_is_taken_in_prompts_not_rows(self):
        """Four distinct prompts over many rows: the split counts the four"""
        prompts = ["a"] * 50 + ["b"] * 50 + ["c"] * 50 + ["d"] * 50
        train, test = _split(prompts, 0.25)
        self.assertEqual({"a", "b", "c"}, {prompts[row] for row in train})
        self.assertEqual({"d"}, {prompts[row] for row in test})

class TestEdges(TestCase):
    """The other half of the method, absent until the weights-only mask was
    shown to rank at 0.938 while generating ' Mary Emma Rose the Rose'."""

    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)

    def test_an_all_open_edge_mask_is_the_model_exactly(self):
        """Not approximately. If the identity is off, every edge number is off."""
        task = build_task("ioi", self.adapter, size=8, seed=0)
        prompts = list(task.clean)
        base = self.adapter.logits(prompts).clone()
        device = next(self.adapter.model.parameters()).device
        gates = {edge: torch.ones((), device=device) for edge in self.adapter.edges()}
        with self.adapter.edge_gate(gates):
            same = self.adapter.logits(prompts)
        self.assertTrue(torch.equal(base, same),
                        "an all-open edge mask changed the model, so the gate is not an identity")

    def test_shutting_every_edge_costs_the_model(self):
        task = build_task("ioi", self.adapter, size=8, seed=0)
        prompts = list(task.clean)
        base = self.adapter.logits(prompts).clone()
        device = next(self.adapter.model.parameters()).device
        gates = {edge: torch.zeros((), device=device) for edge in self.adapter.edges()}
        with self.adapter.edge_gate(gates):
            dead = self.adapter.logits(prompts)
        self.assertGreater(float((dead - base).abs().max()), 1.0,
                           "shutting every edge did not change the logits, so nothing was gated")

    def test_every_enumerated_edge_is_legal(self):
        """`edges()` asks `_check_edge` rather than restating the rule, so the
        two cannot drift; this is the guard on that."""
        for source, destination in self.adapter.edges():
            self.adapter._check_edge(source, destination)

    def test_edges_run_source_before_destination(self):
        for source, destination in self.adapter.edges():
            layer = int(destination.split(":")[1])
            if source == "embed":
                continue
            at = int(source.split(":")[1])
            self.assertLessEqual(at, layer, f"{source} -> {destination} reads forwards in time")

    def test_joint_pruning_learns_both_masks(self):
        task = build_task("ioi", self.adapter, size=32, seed=0)
        sheaf = prune(self.adapter, task, steps=3, batch=8, holdout=0.25, seed=0,
                      sparsity=0.01, edge_sparsity=1.0)
        self.assertEqual(len(self.adapter.edges()), sheaf.n_edges)
        self.assertIsNotNone(sheaf.edge_density)
        self.assertIn("edges", str(sheaf))

    def test_edges_off_leaves_the_weights_only_path_untouched(self):
        """Zero must mean the function is exactly what it was, not a cheap edge run"""
        task = build_task("ioi", self.adapter, size=32, seed=0)
        sheaf = prune(self.adapter, task, steps=3, batch=8, holdout=0.25, seed=0, sparsity=0.01)
        self.assertEqual(0, sheaf.n_edges)
        self.assertIsNone(sheaf.edge_density)
        self.assertNotIn("edges", str(sheaf))

class TestEdgeGradient(TestCase):
    """The receipt the forward tests could not give.

    `tests/discovery.py` checks head_gradients against a finite difference
    because a gradient taken at the wrong site produces a plausible wrong
    ranking. The same trap caught the edge gate: all-open reproduced the model
    exactly and all-shut destroyed it, both passing, while the gradient was 54x
    too large and sign-flipped on an edge into layer 0 -- because the captured
    writes were detached, which severs the path by which closing an early edge
    changes what every later component writes.
    """

    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)

    def _loss(self, task, values, edge_ids, ids, io, subject, device):
        with self.adapter.edge_gate({e: values[i] for i, e in enumerate(edge_ids)}):
            out = self.adapter.model(ids["input_ids"].to(device),
                                     attention_mask=ids["attention_mask"].to(device)).logits
        last = ids["attention_mask"].sum(dim=1).to(device) - 1
        index = torch.arange(out.shape[0], device=device)
        pairs = torch.stack([out[index, last, torch.tensor(io, device=device)],
                             out[index, last, torch.tensor(subject, device=device)]], dim=-1)
        return functional.cross_entropy(
            pairs, torch.zeros(pairs.shape[0], dtype=torch.long, device=device))

    def test_the_edge_gradient_matches_a_finite_difference(self):
        torch.manual_seed(0)
        task = build_task("ioi", self.adapter, size=8, seed=0)
        edge_ids = self.adapter.edges()
        device = next(self.adapter.model.parameters()).device
        io, subject = task.answers(self.adapter)
        ids = self.adapter.tokenizer(list(task.clean), return_tensors="pt", padding=True)

        # An edge into layer 0 -- where the indirect path dominates and the
        # detached version was worst. A late edge would have passed either way.
        pick = next(index for index, (_, destination) in enumerate(edge_ids)
                    if destination == "mlp:0")
        base = torch.full((len(edge_ids),), 0.5, device=device)
        gates = base.clone().requires_grad_(True)
        self._loss(task, [gates[i] for i in range(len(edge_ids))],
                   edge_ids, ids, io, subject, device).backward()
        analytic = float(gates.grad[pick])

        eps = 1e-2
        high, low = base.clone(), base.clone()
        high[pick] += eps
        low[pick] -= eps
        with torch.no_grad():
            difference = float(
                (self._loss(task, [high[i] for i in range(len(edge_ids))],
                            edge_ids, ids, io, subject, device)
                 - self._loss(task, [low[i] for i in range(len(edge_ids))],
                              edge_ids, ids, io, subject, device)) / (2 * eps))
        self.assertGreater(abs(difference), 1e-6, "the finite difference is too small to compare")
        ratio = analytic / difference
        self.assertAlmostEqual(
            1.0, ratio, delta=0.25,
            msg=f"edge gradient is {ratio:.2f}x the finite difference at {edge_ids[pick]}; "
                "the detached-capture bug reads about 54x here")
