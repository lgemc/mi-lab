import random
import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from src.core.metrics import roc_auc
from src.core.probing import (
    LinearProbe,
    ProbeError,
    difference_of_means,
    evaluate,
    measure_scoring_cost,
    train_probe,
)

"""
Probe training is tested on activations this file makes up, not on a
checkpoint: the question here is whether the fitting, the standardization and
the saved artifact are correct, and a real model would only make that slower
to answer.

The fixture puts the signal in a handful of dimensions and buries them in
noise of much larger scale. That is deliberate -- it is the shape of a real
residual stream, and it is the case where a probe that forgets to standardize
looks fine on training data and fails here.
"""

D_MODEL = 32
SIGNAL_DIMS = [3, 17]

def _activations(n: int = 200, seed: int = 0, separation: float = 1.5):
    """Gaussian activations with a linear signal in two dimensions, on wildly different scales"""
    generator = torch.Generator().manual_seed(seed)
    scales = torch.linspace(1.0, 50.0, D_MODEL)
    labels = [index % 2 for index in range(n)]
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    features = torch.randn(n, D_MODEL, generator=generator) * scales
    for dim in SIGNAL_DIMS:
        features[:, dim] += separation * (2 * label_tensor - 1) * scales[dim]
    return features, labels

class TestTrainProbe(TestCase):
    def setUp(self):
        self.train_features, self.train_labels = _activations(n=200, seed=0)
        self.test_features, self.test_labels = _activations(n=200, seed=1)

    def test_it_finds_a_signal_it_has_not_seen(self):
        probe = train_probe(self.train_features, self.train_labels, layer=4, model_id="fixture")
        self.assertGreater(evaluate(probe, self.test_features, self.test_labels)["auc"], 0.95)

    def test_it_puts_its_weight_on_the_signal_dimensions(self):
        probe = train_probe(self.train_features, self.train_labels, layer=4, model_id="fixture")
        ranked = probe.weight.abs().argsort(descending=True)[: len(SIGNAL_DIMS)]
        self.assertEqual(set(SIGNAL_DIMS), set(int(dim) for dim in ranked))

    def test_labels_that_mean_nothing_do_not_produce_a_usable_probe(self):
        """The control that catches evaluation leakage

        Not asserted to land exactly at chance: with 200 examples a random
        label vector correlates with the signal dimensions by a few percent
        just by luck, and the test set amplifies that because those same
        dimensions are what it is separable on. The claim that holds is the
        one that matters -- a probe fit on meaningless labels is nowhere near
        a probe fit on real ones.
        """
        shuffled = list(self.train_labels)
        random.Random(11).shuffle(shuffled)
        noise = train_probe(self.train_features, shuffled, layer=4, model_id="fixture", seed=3)
        signal = train_probe(self.train_features, self.train_labels, layer=4, model_id="fixture", seed=3)

        noise_auc = evaluate(noise, self.test_features, self.test_labels)["auc"]
        signal_auc = evaluate(signal, self.test_features, self.test_labels)["auc"]
        self.assertLess(noise_auc, 0.8)
        self.assertGreater(signal_auc - noise_auc, 0.2)

    def test_the_same_seed_gives_the_same_probe(self):
        first = train_probe(self.train_features, self.train_labels, layer=4, model_id="fixture", seed=5)
        second = train_probe(self.train_features, self.train_labels, layer=4, model_id="fixture", seed=5)
        self.assertTrue(torch.equal(first.weight, second.weight))

    def test_one_class_is_refused(self):
        with self.assertRaises(ProbeError):
            train_probe(self.train_features, [1] * 200, layer=4, model_id="fixture")

    def test_mismatched_labels_are_refused(self):
        with self.assertRaises(ProbeError):
            train_probe(self.train_features, self.train_labels[:10], layer=4, model_id="fixture")

class TestDifferenceOfMeans(TestCase):
    def test_the_untrained_baseline_also_finds_a_clean_signal(self):
        train_features, train_labels = _activations(n=200, seed=0)
        test_features, test_labels = _activations(n=200, seed=1)
        probe = difference_of_means(train_features, train_labels, layer=4, model_id="fixture")
        self.assertGreater(evaluate(probe, test_features, test_labels)["auc"], 0.9)
        self.assertEqual("difference_of_means", probe.method)

    def test_its_direction_is_a_unit_vector(self):
        features, labels = _activations(n=100, seed=2)
        probe = difference_of_means(features, labels, layer=4, model_id="fixture")
        self.assertAlmostEqual(1.0, float(probe.weight.norm()), places=5)

class TestScoring(TestCase):
    def setUp(self):
        features, labels = _activations(n=100, seed=0)
        self.features, self.labels = features, labels
        self.probe = train_probe(features, labels, layer=4, model_id="fixture")

    def test_a_layer_axis_of_one_is_accepted(self):
        flat = self.probe.score(self.features)
        with_layer = self.probe.score(self.features[:, None, :])
        self.assertTrue(torch.equal(flat, with_layer))

    def test_activations_of_the_wrong_width_are_refused(self):
        with self.assertRaises(ProbeError):
            self.probe.score(torch.randn(4, D_MODEL + 1))

    def test_scores_are_consistent_with_the_reported_auc(self):
        metrics = evaluate(self.probe, self.features, self.labels)
        self.assertAlmostEqual(metrics["auc"], roc_auc(self.probe.score(self.features), self.labels))

class TestArtifact(TestCase):
    def setUp(self):
        features, labels = _activations(n=100, seed=0)
        self.features = features
        self.probe = train_probe(features, labels, layer=4, model_id="fixture", dataset="fixture-set")

    def test_a_saved_probe_scores_identically_when_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "probe.pt")
            self.probe.save(path)
            loaded = LinearProbe.load(path)
            self.assertTrue(torch.allclose(self.probe.score(self.features), loaded.score(self.features)))

    def test_provenance_survives_the_round_trip(self):
        """A direction without its layer and model cannot be applied to anything"""
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "probe.pt")
            self.probe.save(path)
            loaded = LinearProbe.load(path)
            self.assertEqual((4, "fixture", "fixture-set"), (loaded.layer, loaded.model_id, loaded.dataset))

    def test_a_missing_probe_is_refused(self):
        with self.assertRaises(ProbeError):
            LinearProbe.load("/tmp/definitely-not-a-probe.pt")

    def test_the_artifact_is_kilobytes(self):
        self.assertEqual(3 * D_MODEL * 8, self.probe.n_bytes)

    def test_scoring_cost_is_reported_per_activation(self):
        cost = measure_scoring_cost(self.probe, self.features, repeats=5)
        self.assertEqual(500, cost.items)
        self.assertGreater(cost.ms_per_item, 0.0)
