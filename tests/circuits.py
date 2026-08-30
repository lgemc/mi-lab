from unittest import TestCase

import torch

from src.core.config import ConfigError, Position
from src.core.metrics import logit_difference
from src.data.ioi import build_ioi
from src.methods.circuits import (
    Circuit,
    CircuitError,
    baselines,
    direct_logit_attribution,
    patch_heads,
    verify,
)
from src.model.adapter import require_circuits

from .stubs.model import shared_adapter

"""
Circuit tests need a real checkpoint and skip loudly without one.

Two of them are the load-bearing ones. The first is that the decomposition
adds up: every write into the residual stream, summed and pushed through the
frozen unembedding, has to land on the logit difference the model actually
produced. If that drifts, every attribution number is wrong by an amount
nothing else would report.

The second is that patching is a no-op when it writes back what was already
there. Every causal number in this package is a difference against that, so a
patch mechanism that perturbs even slightly on its own turns the entire study
into an artefact of the hooks.

The replication check is deliberately loose about magnitudes and strict about
identity: GPT-2 small's strongest direct writer towards the indirect object is
head 9.9 and its strongest writer against it is 10.7. Those are the paper's
name mover and negative name mover, and if this framework stops finding them
it has stopped measuring the thing it claims to.
"""

NAME_MOVER = (9, 9)
NEGATIVE_NAME_MOVER = (10, 7)

class CircuitTestCase(TestCase):
    adapter = None
    dataset = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)
        cls.dataset = build_ioi(cls.adapter, size=8, seed=0)

class TestDecomposition(CircuitTestCase):
    def test_the_parts_add_up_to_the_measured_answer(self):
        """The receipt on every attribution number in this package"""
        result = direct_logit_attribution(self.adapter, self.dataset)
        self.assertLess(abs(result.residual), 1e-3, f"{abs(result.residual):.2e} of the answer went unattributed")

    def test_the_residual_stream_is_the_sum_of_its_writes(self):
        decomposition = self.adapter.decompose(self.dataset.clean[:2])
        drift = decomposition.remainder.abs().max().item()
        self.assertLess(drift, 1e-2, f"the split misses {drift:.2e} of the residual stream")

    def test_it_finds_the_paper_s_name_movers(self):
        result = direct_logit_attribution(self.adapter, self.dataset)
        self.assertIn(NAME_MOVER, [head for head, _ in result.top(3)])
        self.assertIn(NEGATIVE_NAME_MOVER, [head for head, _ in result.top(3, negative=True)])

class TestPatching(CircuitTestCase):
    def test_writing_back_what_was_there_changes_nothing(self):
        """Every causal number is a difference against this being exactly zero"""
        before = self.adapter.logits(self.dataset.clean)
        donors = self.adapter.head_outputs(self.dataset.clean, layers=[9])
        with self.adapter.patch(heads={9: {head: donors[:, 0, head] for head in range(self.adapter.cfg.n_heads)}}):
            after = self.adapter.logits(self.dataset.clean)
        self.assertTrue(torch.allclose(before, after, atol=1e-4))

    def test_restoring_every_layer_restores_the_clean_run(self):
        clean = self.adapter.logits(self.dataset.clean)
        layers = list(range(self.adapter.cfg.n_layers))
        donor = self.adapter.capture(self.dataset.clean, layers=layers, position=Position.ALL)
        with self.adapter.patch(residual={layer: donor[:, layer] for layer in layers}):
            patched = self.adapter.logits(self.dataset.corrupted)
        self.assertTrue(torch.allclose(clean, patched, atol=1e-3))

    def test_a_patch_survives_being_split_across_batches(self):
        """The donor covers every prompt; the hooks have to slice the chunk they fire in"""
        wide = self.adapter.logits(self.dataset.corrupted)
        donors = self.adapter.head_outputs(self.dataset.clean, layers=[9])
        with self.adapter.patch(heads={9: {9: donors[:, 0, 9]}}):
            in_one_batch = self.adapter.logits(self.dataset.corrupted)

        original = self.adapter.cfg
        self.adapter.cfg = original.__class__(**{**original.as_dict(), "batch_size": 2, "sae": None})
        try:
            with self.adapter.patch(heads={9: {9: donors[:, 0, 9]}}):
                in_four_batches = self.adapter.logits(self.dataset.corrupted)
        finally:
            self.adapter.cfg = original

        self.assertFalse(torch.allclose(wide, in_one_batch, atol=1e-3))
        self.assertTrue(torch.allclose(in_one_batch, in_four_batches, atol=1e-3))

    def test_patches_do_not_nest(self):
        donor = self.adapter.head_outputs(self.dataset.clean, layers=[9])[:, 0, 9]
        with (
            self.adapter.patch(heads={9: {9: donor}}),
            self.assertRaises(ConfigError),
            self.adapter.patch(residual={0: torch.zeros(1)}),
        ):
            pass

    def test_head_effects_name_the_layers_they_swept(self):
        """A partial sweep still has a row 0, and calling it layer 0 would be a lie"""
        effects = patch_heads(self.adapter, self.dataset, layers=[9, 10])
        self.assertEqual(effects.layers, [9, 10])
        self.assertEqual({layer for (layer, _), _ in effects.ranked()}, {9, 10})

class TestBaselines(CircuitTestCase):
    def test_the_span_is_what_the_corruption_opened(self):
        reference = baselines(self.adapter, self.dataset)
        self.assertGreater(reference.span, 0.5)
        self.assertAlmostEqual(reference.recovery(reference.clean), 1.0, places=5)
        self.assertAlmostEqual(reference.recovery(reference.corrupted), 0.0, places=5)

    def test_a_corruption_that_corrupts_nothing_is_refused(self):
        dataset = build_ioi(self.adapter, size=4, seed=0)
        for example in list(dataset.examples):
            dataset.examples[dataset.examples.index(example)] = example.__class__(
                clean=example.clean, corrupted=example.clean, io=example.io,
                subject=example.subject, order=example.order,
            )
        with self.assertRaises(CircuitError) as caught:
            baselines(self.adapter, dataset)
        self.assertIn("span", str(caught.exception))

    def test_the_model_does_the_task_at_all(self):
        io, subject = self.dataset.answers(self.adapter)
        differences = logit_difference(self.adapter.logits(self.dataset.clean), io, subject)
        self.assertGreater(float((differences > 0).double().mean()), 0.7)

class TestVerification(CircuitTestCase):
    def test_an_empty_circuit_cannot_be_verified(self):
        with self.assertRaises(CircuitError):
            verify(self.adapter, self.dataset, Circuit(heads=[]))

    def test_the_name_mover_alone_recovers_something(self):
        report = verify(self.adapter, self.dataset, Circuit(heads=[NAME_MOVER]))
        self.assertGreater(report.faithfulness, 0.0)
        self.assertEqual(list(report.minimality), [NAME_MOVER])
