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
from src.methods.sheaves import gumbel_sigmoid, prune, schedule

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
