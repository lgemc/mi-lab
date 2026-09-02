"""Guards on the neuron scan: a contrast, a threshold, and where the flagged ones sit.

The synthetic half plants known neurons in a [n_layers, d_ff] score and checks
that the contrast, the sigma flag, the top list, the concentration split and
the survival under a second contrast all find exactly them. The online half
runs the real hooks on GPT-2 small: a mean over a few texts has to have the
MLP's width, and a trace has to line up token for token with its text.
"""

from unittest import TestCase

import torch

from src.methods.neurons import (
    NeuronError,
    concentration,
    contrast,
    control_texts,
    flag,
    mean_abs_activation,
    per_layer_counts,
    summarize,
    survival,
    top_neurons,
    trace,
)

from .stubs.model import shared_adapter


def planted() -> torch.Tensor:
    """8 layers of 16 neurons, quiet everywhere except three neurons that stand out"""
    torch.manual_seed(0)
    score = torch.randn(8, 16) * 0.01
    score[7, 3] = 5.0
    score[0, 9] = 4.0
    score[4, 1] = 3.0
    return score


class TestControlTexts(TestCase):
    def test_they_alternate_math_and_code_and_carry_no_prose(self):
        lines = control_texts(4)
        self.assertEqual(4, len(lines))
        self.assertIn("=", lines[0])
        self.assertTrue(lines[1].startswith("def f1("))
        self.assertEqual(lines, control_texts(4))
        self.assertNotEqual(lines, control_texts(4, seed=1))


class TestContrast(TestCase):
    def test_the_score_is_target_minus_the_mean_of_the_others(self):
        target = torch.full((2, 3), 4.0)
        others = [torch.full((2, 3), 1.0), torch.full((2, 3), 3.0)]
        self.assertTrue(torch.equal(torch.full((2, 3), 2.0), contrast(target, others)))

    def test_no_condition_or_a_different_shape_is_refused(self):
        with self.assertRaises(NeuronError):
            contrast(torch.zeros(2, 3), [])
        with self.assertRaises(NeuronError):
            contrast(torch.zeros(2, 3), [torch.zeros(3, 2)])


class TestFlagging(TestCase):
    def test_the_planted_neurons_are_the_flagged_ones(self):
        flagged, sigma = flag(planted())
        self.assertGreater(sigma, 0.0)
        self.assertEqual({(7, 3), (0, 9), (4, 1)}, {tuple(index.tolist()) for index in flagged.nonzero()})

    def test_the_top_list_is_best_first_with_layer_and_neuron(self):
        top = top_neurons(planted(), 2)
        self.assertEqual([(7, 3, 5.0), (0, 9, 4.0)], [(row["layer"], row["neuron"], row["score"]) for row in top])

    def test_the_top_list_cannot_exceed_the_neuron_count(self):
        self.assertEqual(6, len(top_neurons(torch.zeros(2, 3), 50)))

    def test_concentration_splits_the_stack_into_quarters(self):
        flagged, _ = flag(planted())
        split = concentration(flagged)
        self.assertEqual(2, split["bottom_quarter_layers"])
        self.assertEqual(1, split["bottom_quarter_flagged"])
        self.assertEqual(1, split["top_quarter_flagged"])
        self.assertEqual(1, split["middle_flagged"])
        self.assertAlmostEqual(2 / 3, split["bottom_plus_top_share"], places=3)

    def test_nothing_flagged_has_no_share(self):
        self.assertIsNone(concentration(torch.zeros(4, 4, dtype=torch.bool))["bottom_plus_top_share"])

    def test_per_layer_counts_are_keyed_as_strings_and_skip_empty_layers(self):
        flagged, _ = flag(planted())
        self.assertEqual({"0": 1, "4": 1, "7": 1}, per_layer_counts(flagged))


class TestSurvival(TestCase):
    def test_a_neuron_that_vanishes_under_the_second_contrast_says_so(self):
        earlier = top_neurons(planted(), 3)
        second = planted()
        second[4, 1] = 0.0                       # gone under the second condition
        rows = survival(earlier, second, count=2)
        by_key = {(row["layer"], row["neuron"]): row for row in rows}
        self.assertTrue(by_key[(7, 3)]["in_second_top"])
        self.assertTrue(by_key[(7, 3)]["flagged_second"])
        self.assertFalse(by_key[(4, 1)]["in_second_top"])
        self.assertFalse(by_key[(4, 1)]["flagged_second"])
        self.assertEqual(0.0, by_key[(4, 1)]["second_score"])
        # the earlier fields ride along untouched
        self.assertEqual(5.0, by_key[(7, 3)]["score"])


class TestOnline(TestCase):
    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = adapter

    def test_the_mean_activation_has_one_row_per_layer_and_the_mlp_width(self):
        texts = ["The quick brown fox jumps over the lazy dog.", "Another short sentence for the scan."]
        mean, tokens = mean_abs_activation(self.adapter, texts, minimum_tokens=4)
        self.assertEqual((self.adapter.cfg.n_layers, 4 * self.adapter.cfg.d_model), tuple(mean.shape))
        self.assertGreater(tokens, 0)
        self.assertTrue(bool((mean >= 0).all()))

    def test_a_trace_lines_up_with_its_tokens(self):
        texts = ["Hola mundo, esto es una prueba."]
        (activations, tokens), = trace(self.adapter, texts, layer=3, neurons=[0, 7])
        self.assertEqual((len(tokens), 2), tuple(activations.shape))
        self.assertEqual("".join(tokens), texts[0])
        report = summarize([(activations, tokens)], column=1, top=3)
        self.assertEqual(3, len(report["top_firing"]))
        self.assertIn(report["top_firing"][0]["token"], tokens)

    def test_summarizing_nothing_is_refused(self):
        with self.assertRaises(NeuronError):
            summarize([], column=0)
