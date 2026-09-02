"""The receipts on edge ablation, because a plausible wrong graph is the failure mode.

An edge intervention is arithmetic on a sum, so it is only as good as the claim
that the sum reproduces the forward pass. Two of these tests exist because they
failed: the embedding had no live value to subtract, and reconstructing a head's
write re-entered the hook capturing that projection's input. Both produced
numbers rather than errors.
"""

from unittest import TestCase

import torch

from src.core.config import ConfigError
from src.data.tasks import build_task
from src.methods.circuits import require_circuits
from src.model.backends.transformers import _attention_norm

from .stubs.model import shared_adapter

DESTINATION = 5

class EdgeTestCase(TestCase):
    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)
        # length-aligned by construction, which an edge needs: it is a term in one
        # position's residual sum, so both runs must have the same positions
        task = build_task("ioi", cls.adapter, size=4, seed=0)
        cls.clean, cls.corrupted = list(task.clean), list(task.corrupted)

    def upstream(self, layer=DESTINATION):
        return (["embed"]
                + [f"head:{at}:{head}" for at in range(layer)
                   for head in range(self.adapter.cfg.n_heads)]
                + [f"bias:{at}" for at in range(layer)] + [f"mlp:{at}" for at in range(layer)])

class TestDecomposition(EdgeTestCase):
    def test_the_writes_sum_to_what_the_destinations_read(self):
        """Nothing built on this graph means anything if it is not the model's graph"""
        sources = self.adapter.residual_sources(self.clean)
        gap = self.adapter.residual_remainder(sources)
        self.assertLess(gap["relative"], 1e-4, "the residual decomposition does not reproduce the forward pass")

class TestEdgePatch(EdgeTestCase):
    def test_ablating_no_edges_is_byte_identical(self):
        off = self.adapter.residual_sources(self.corrupted)
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, []):
            self.assertTrue(torch.equal(base, self.adapter.logits(self.clean)))

    def test_ablating_toward_the_run_itself_changes_nothing(self):
        """The identity that caught both bugs: a no-op written the long way round

        Ablating every edge toward the values the run already has must be a
        no-op. It was not, twice -- once because `embed` had no live write to
        subtract and its counterfactual landed on top of the real one, once
        because reconstructing a head re-entered the pre-hook capturing that
        projection's input and left every later head reading zeros. Both moved
        the logits by more than two.
        """
        itself = self.adapter.residual_sources(self.clean)
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(itself, [(s, f"attn:{DESTINATION}") for s in self.upstream()]):
            patched = self.adapter.logits(self.clean)
        self.assertLess(float((patched - base).abs().max()), 1e-2)

    def test_every_edge_into_a_destination_equals_replacing_its_input(self):
        """Edge ablation has to agree with the coarser thing it refines"""
        off = self.adapter.residual_sources(self.corrupted)
        with self.adapter.edge_patch(off, [(s, f"attn:{DESTINATION}") for s in self.upstream()]):
            via_edges = self.adapter.logits(self.clean)
        donor = off["attention_in"][:, DESTINATION]
        handle = _attention_norm(self.adapter.blocks[DESTINATION], DESTINATION).register_forward_pre_hook(
            lambda module, args: (donor.to(args[0].device, args[0].dtype), *args[1:])
        )
        try:
            via_replacement = self.adapter.logits(self.clean)
        finally:
            handle.remove()
        self.assertLess(float((via_edges - via_replacement).abs().max()), 1e-2)

    def test_one_edge_moves_the_answer(self):
        off = self.adapter.residual_sources(self.corrupted)
        base = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, [("head:0:0", f"attn:{DESTINATION}")]):
            self.assertFalse(torch.equal(base, self.adapter.logits(self.clean)))

    def test_edges_compose_rather_than_overwrite(self):
        """Two edges must differ from either alone, or later edits are ignoring earlier ones"""
        off = self.adapter.residual_sources(self.corrupted)
        with self.adapter.edge_patch(off, [("head:0:0", "attn:5")]):
            first = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, [("head:1:0", "attn:5")]):
            second = self.adapter.logits(self.clean)
        with self.adapter.edge_patch(off, [("head:0:0", "attn:5"), ("head:1:0", "attn:5")]):
            both = self.adapter.logits(self.clean)
        self.assertFalse(torch.equal(both, first))
        self.assertFalse(torch.equal(both, second))

    def test_a_non_causal_edge_is_refused(self):
        """An edge that does not exist must raise, not quietly ablate nothing"""
        off = self.adapter.residual_sources(self.corrupted)
        for edge in (("mlp:5", "attn:5"), ("head:7:0", "attn:3"), ("mlp:9", "mlp:2")):
            with self.subTest(edge=edge), self.assertRaises(ConfigError), \
                    self.adapter.edge_patch(off, [edge]):
                pass

    def test_runs_of_different_lengths_are_refused(self):
        """A mixture of positions is a mixture of different tokens"""
        off = self.adapter.residual_sources(self.corrupted)
        with self.assertRaises(ConfigError), self.adapter.edge_patch(off, [("head:0:0", "attn:5")]):
            self.adapter.logits(["a much shorter prompt"])
