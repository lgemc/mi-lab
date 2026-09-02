"""Guards on the pruner, most of which exist because a number looked too good.

A weight mask has one gate per parameter -- 85M of them on GPT-2 small -- and
whatever it is scored on, it can memorize. The first run of this reported 0.02%
of weights open at accuracy 1.000, which was twenty thousand weights having
learned eight sentences. These tests are what stops that being reported again.
"""

from unittest import TestCase

import torch

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
)

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
