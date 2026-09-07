from unittest import TestCase

import torch

from src.core.metrics import spearman
from src.core.readout import ReadoutError
from src.data.tasks import build_task
from src.methods.circuits.techniques import TECHNIQUES, DiscoveryError, rank, technique_names
from src.model.adapter import require_circuits

from ..stubs.model import shared_adapter

"""
Discovery tests need a real checkpoint and skip loudly without one.

Two of them are load-bearing. The first is that `gradients` is the gradient of
the thing it says it is: perturb one head's output by a small amount, and the
change in the readout has to be the one the gradient predicted. Everything eap
reports is that inner product, so a gradient taken at the wrong site, or with
the graph cut, produces a ranking that looks entirely plausible and is
answering a different question.

Its expected values did not move when the measurement contract landed, and
that is the point of it being here: `gradients(readout=)` differentiates the
same quantity `head_gradients` did, and this is what says so.

The second is that eap tracks patching. It is the claim the technique is used
for -- one backward pass standing in for a forward pass per head -- and it is
the claim that stops being true first when something upstream changes. It is
asserted relatively as well as absolutely: eap has to agree with patching more
than direct attribution does, because if it did not there would be no reason
to run it over the cheaper thing.

The expensive techniques are swept over four layers rather than twelve. The
question they are being asked here is whether they work, not what GPT-2 does.
"""

LAYERS = [8, 9, 10, 11]

class _NoGradient:
    """A well-formed Score that says it has no gradient, standing in for BLEU"""
    name = "no-gradient"
    units = "share"
    definition = "a score with no derivative, for testing the refusal"
    differentiable = False

    def __call__(self, output):
        return output.sum(dim=-1).detach()

    def select(self, rows):
        return self

class DiscoveryTestCase(TestCase):
    adapter = None
    task = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)
        cls.task = build_task("ioi", cls.adapter, size=4, seed=0)

class TestGradients(DiscoveryTestCase):
    def test_the_gradient_predicts_what_a_small_change_does(self):
        """The receipt on every number eap reports"""
        score = self.task.readout(self.adapter)
        gradients = self.adapter.gradients(self.task.clean, score, layers=[9])
        donors = self.adapter.head_outputs(self.task.clean, layers=[9])

        before = float(score(self.adapter.outputs(self.task.clean)).mean())
        step = torch.zeros_like(donors[:, 0, 9])
        step[..., :] = 1e-3
        with self.adapter.patch(heads={9: {9: donors[:, 0, 9] + step}}):
            after = float(score(self.adapter.outputs(self.task.clean)).mean())
        predicted = float((gradients[:, 0, 9] * step).sum(dim=(1, 2)).mean())
        self.assertAlmostEqual(after - before, predicted, delta=max(2e-2 * abs(predicted), 1e-3))

    def test_the_ids_have_to_index_the_prompts(self):
        """A readout built for one batch, scored against another, is refused rather than broadcast"""
        score = self.task.readout(self.adapter)
        with self.assertRaises(ValueError):
            self.adapter.gradients(self.task.clean, score.select(slice(0, 1)), layers=[9])

    def test_a_gradient_refuses_a_readout_it_cannot_differentiate(self):
        """The refusal lands before the forward pass, not as a missing grad_fn inside it"""
        with self.assertRaises(ReadoutError) as raised:
            self.adapter.gradients(self.task.clean, _NoGradient(), layers=[9])
        self.assertIn("gradient through the score", str(raised.exception))

class TestTechniques(DiscoveryTestCase):
    def test_every_technique_scores_every_head_it_swept(self):
        for name in technique_names():
            with self.subTest(technique=name):
                ranking = rank(name, self.adapter, self.task, layers=LAYERS)
                self.assertEqual(tuple(ranking.scores.shape), (len(LAYERS), self.adapter.cfg.n_heads))
                self.assertEqual(ranking.layers, LAYERS)
                self.assertEqual(ranking.units, TECHNIQUES[name].units)

    def test_a_ranking_names_the_layers_it_swept(self):
        """A sweep over part of the model still has a row 0, and it is not layer 0"""
        ranking = rank("eap", self.adapter, self.task, layers=LAYERS)
        self.assertTrue(all(layer in LAYERS for (layer, _), _ in ranking.ranked(5)))

    def test_eap_tracks_patching(self):
        patching = rank("patching", self.adapter, self.task, layers=LAYERS)
        eap = rank("eap", self.adapter, self.task, layers=LAYERS)
        self.assertGreater(spearman(patching.flat(), eap.flat()), 0.8)

    def test_eap_tracks_patching_better_than_the_direct_path_does(self):
        """If it did not, there would be no reason to run it over the cheaper thing"""
        patching = rank("patching", self.adapter, self.task, layers=LAYERS)
        eap = rank("eap", self.adapter, self.task, layers=LAYERS)
        attribution = rank("attribution", self.adapter, self.task, layers=LAYERS)
        self.assertGreater(
            spearman(patching.flat(), eap.flat()), spearman(patching.flat(), attribution.flat())
        )

    def test_eap_costs_the_same_whatever_the_model(self):
        wide = rank("eap", self.adapter, self.task)
        narrow = rank("eap", self.adapter, self.task, layers=LAYERS)
        self.assertEqual(wide.passes, narrow.passes)
        self.assertGreater(rank("patching", self.adapter, self.task, layers=LAYERS).passes, wide.passes)

    def test_the_random_control_is_reproducible_and_free(self):
        first = rank("random", self.adapter, self.task, layers=LAYERS, seed=7)
        second = rank("random", self.adapter, self.task, layers=LAYERS, seed=7)
        self.assertTrue(torch.equal(first.scores, second.scores))
        self.assertEqual(first.passes, 0)

class TestSelection(DiscoveryTestCase):
    def setUp(self):
        self.ranking = rank("random", self.adapter, self.task, layers=LAYERS, seed=0)

    def test_a_circuit_is_the_top_heads_by_absolute_score(self):
        circuit = self.ranking.select(count=5)
        self.assertEqual(len(circuit), 5)
        self.assertEqual(circuit.heads, [head for head, _ in self.ranking.ranked(5)])

    def test_a_share_is_a_share_of_the_heads_ranked(self):
        circuit = self.ranking.select(frac=0.1)
        self.assertEqual(len(circuit), round(0.1 * len(LAYERS) * self.adapter.cfg.n_heads))

    def test_a_selected_circuit_carries_no_growth_curve(self):
        """That field is the greedy search's cumulative recovery, and this set never walked one"""
        self.assertEqual(self.ranking.select(count=3).scores, [])

    def test_count_and_frac_are_exclusive(self):
        with self.assertRaises(DiscoveryError):
            self.ranking.select(count=3, frac=0.1)
        with self.assertRaises(DiscoveryError):
            self.ranking.select()

    def test_a_circuit_bigger_than_the_sweep_is_refused(self):
        with self.assertRaises(DiscoveryError):
            self.ranking.select(count=10_000)

    def test_an_unknown_technique_names_the_ones_that_exist(self):
        with self.assertRaises(DiscoveryError) as raised:
            rank("edge_patching", self.adapter, self.task)
        self.assertIn("patching", str(raised.exception))
