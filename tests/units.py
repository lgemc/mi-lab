"""Guards on the unit gates and the attribution warm start.

A unit gate is one parameter that closes a whole head, neuron or block, and
the whole point is that it closes *every* weight of that unit in every tensor
the unit touches: a head whose q/k/v are closed but whose output rows are
still open is not a closed head, and a density counted on the weight gates
alone would not notice. These tests pin the bindings to the layouts GPT-2 and
Qwen3 actually use, and check that closing a unit reaches the weights.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from src.data.tasks import build_task
from src.methods.sheaves.gateable import gateable
from src.methods.sheaves.training import prune
from src.methods.sheaves.units import Units, bindings, init_from, rank_logits, unit_scores
from src.telemetry.journal import Journal, read_metrics

from .stubs.model import shared_adapter


class TestRanking(TestCase):
    def test_ranks_span_the_bounds_in_score_order(self):
        score = torch.tensor([[0.3, 0.1], [0.9, 0.2]])
        logits = rank_logits(score, 2.0, 5.0)
        self.assertEqual(logits.shape, score.shape)
        self.assertAlmostEqual(float(logits.min()), 2.0)
        self.assertAlmostEqual(float(logits.max()), 5.0)
        self.assertEqual(int(logits.argmax()), int(score.argmax()))
        self.assertEqual(int(logits.argmin()), int(score.argmin()))
        self.assertTrue(torch.equal(logits.flatten().argsort(), score.flatten().argsort()))

    def test_a_single_element_ranks_at_the_top(self):
        self.assertAlmostEqual(float(rank_logits(torch.tensor([7.0]), 2.0, 5.0)), 5.0)


class TestBindings(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available")
        cls.adapter = adapter
        cls.shapes = {name: tuple(p.shape) for name, p in gateable(adapter, [3]).items()}

    def bound(self, leaf):
        name = next(n for n in self.shapes if n.endswith(leaf))
        return name, {b.unit: b for b in bindings(self.adapter, name, self.shapes[name])}

    def test_gpt2_fused_qkv_binds_twelve_heads_three_times_along_the_output_axis(self):
        _, bound = self.bound("attn.c_attn.weight")
        head = bound["head:3"]
        self.assertEqual(head.axis, 1)
        self.assertEqual(head.n_units, 12)
        self.assertEqual(head.slots.numel(), 2304)
        # q, k and v of head 0 are the first 64 columns of each third
        for third in range(3):
            self.assertEqual(int(head.slots[third * 768]), 0)
            self.assertEqual(int(head.slots[third * 768 + 64]), 1)
        self.assertEqual(bound["block:3:attn"].axis, 0)

    def test_the_attention_output_binds_heads_on_its_input_axis(self):
        _, bound = self.bound("attn.c_proj.weight")
        self.assertEqual(bound["head:3"].axis, 0)
        self.assertEqual(bound["head:3"].n_units, 12)
        _, bias = self.bound("attn.c_proj.bias")
        self.assertEqual(set(bias), {"block:3:attn"}, "an output bias is nobody's head")

    def test_mlp_neurons_bind_the_hidden_axis_on_both_sides(self):
        _, up = self.bound("mlp.c_fc.weight")
        self.assertEqual((up["neuron:3"].axis, up["neuron:3"].n_units), (1, 3072))
        _, down = self.bound("mlp.c_proj.weight")
        self.assertEqual((down["neuron:3"].axis, down["neuron:3"].n_units), (0, 3072))
        _, bias = self.bound("mlp.c_fc.bias")
        self.assertEqual(bias["neuron:3"].axis, 0, "the up-projection bias is per neuron")

    def test_a_closed_unit_closes_every_weight_it_touches(self):
        units = Units.build(self.adapter, self.shapes, 5.0, torch.device("cpu"),
                            families=("head", "neuron", "block"))
        with torch.no_grad():
            units.logits["head:3"][4] = -5.0
        for leaf, axis in (("attn.c_attn.weight", 1), ("attn.c_proj.weight", 0)):
            name = next(n for n in self.shapes if n.endswith(leaf))
            hard = units.hard(name, 2).expand(self.shapes[name])
            closed = ~hard
            self.assertEqual(int(closed.sum()), 3 * 768 * 64 if axis == 1 else 64 * 768)
            self.assertTrue(bool(hard.select(axis, 0).all()), "head 0 stayed open")
        with torch.no_grad():
            units.logits["block:3:mlp"][0] = -5.0
        for leaf in ("mlp.c_fc.weight", "mlp.c_proj.weight", "mlp.c_proj.bias"):
            name = next(n for n in self.shapes if n.endswith(leaf))
            self.assertFalse(bool(units.hard(name, len(self.shapes[name])).any()))
        counts = units.counts()
        self.assertEqual(counts["head"], {"open": 11, "total": 12})
        self.assertEqual(counts["block"], {"open": 1, "total": 2})
        self.assertEqual(units.report()["blocks_closed"], ["block:3:mlp"])

    def test_the_default_families_leave_blocks_ungated(self):
        units = Units.build(self.adapter, self.shapes, 5.0, torch.device("cpu"))
        self.assertEqual(set(units.logits), {"head:3", "neuron:3"})
        with self.assertRaises(ValueError):
            Units.build(self.adapter, self.shapes, 5.0, torch.device("cpu"), families=("layer",))

    def test_the_drawn_factor_agrees_with_the_hard_product(self):
        units = Units.build(self.adapter, self.shapes, 5.0, torch.device("cpu"))
        with torch.no_grad():
            units.logits["neuron:3"][:100] = -5.0
        name = next(n for n in self.shapes if n.endswith("mlp.c_proj.weight"))
        with units.draw(lambda logits: (logits > 0).float()):
            drawn = units.factor(name, 2, deterministic=False)
        self.assertTrue(torch.equal(drawn.bool(), units.hard(name, 2)))
        self.assertEqual(int(drawn.sum()), 3072 - 100)

    def test_attribution_ranks_the_implicated_weights_and_units_higher(self):
        units = Units.build(self.adapter, self.shapes, 5.0, torch.device("cpu"))
        gates = {name: torch.zeros(shape) for name, shape in self.shapes.items()}
        scores = {name: torch.rand(shape) for name, shape in self.shapes.items()}
        name = next(n for n in self.shapes if n.endswith("attn.c_proj.weight"))
        scores[name][64 * 7:64 * 8] += 10.0  # head 7 of the output projection
        init_from(scores, gates, units, 2.0, 5.0)
        for name, logits in gates.items():
            self.assertGreaterEqual(float(logits.min()), 2.0)
            self.assertLessEqual(float(logits.max()), 5.0)
            self.assertEqual(int(logits.argmax()), int(scores[name].argmax()))
        totals = unit_scores(units, scores)
        self.assertEqual(int(totals["head:3"].argmax()), 7)
        self.assertEqual(int(units.logits["head:3"].argmax()), 7)
        self.assertAlmostEqual(float(units.logits["head:3"][7].detach()), 5.0)


class TestGranularPruning(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available")
        cls.adapter = adapter

    def test_a_granular_run_returns_the_effective_mask_and_a_unit_report(self):
        task = build_task("ioi", self.adapter, size=8, seed=0)
        with TemporaryDirectory() as directory:
            journal = Journal(Path(directory) / "run", params={})
            sheaf = prune(self.adapter, task, steps=3, batch=4, holdout=0.25, journal=journal,
                          probe_every=1, faith_kind="nll", sparsity=0.0, max_times=1.0,
                          layers=[2], init=5.0, granular=["head", "neuron", "block"],
                          attribute=1, init_low=2.0)
            journal.finish()
            rows = read_metrics(Path(directory) / "run")
        self.assertTrue(all(mask.dtype == torch.bool for mask in sheaf.gates.values()))
        self.assertEqual(sheaf.n_open, sum(int(mask.sum()) for mask in sheaf.gates.values()))
        self.assertIsNotNone(sheaf.units)
        self.assertEqual(sheaf.units["counts"]["head"]["total"], 12)
        self.assertEqual(sheaf.units["counts"]["neuron"]["total"], 3072)
        self.assertEqual(sheaf.units["counts"]["block"]["total"], 2)
        self.assertEqual(set(sheaf.unit_logits), {"head:2", "neuron:2", "block:2:attn", "block:2:mlp"})
        self.assertTrue(any("heads_open" in row for row in rows))
        self.assertIn("head", str(sheaf))
