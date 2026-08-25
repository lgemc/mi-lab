from unittest import TestCase

import torch

from src.core.metrics import (
    Cost,
    MetricError,
    accuracy,
    best_threshold,
    logit_difference,
    measure,
    recovery,
    roc_auc,
)

"""
AUC is the number every later result is quoted in, so it is worth pinning
against hand-computable cases: perfect separation, perfect inversion, and the
all-ties case where the answer must be exactly a coin flip.

The circuit metrics get the same treatment. recovery in particular has one
case that has to be a decision rather than an accident: a clean and corrupted
baseline that coincide leave no span to normalize against, and the answer has
to be zero rather than an infinity that propagates into a heatmap.
"""

class TestRocAuc(TestCase):
    def test_perfect_separation_is_one(self):
        self.assertAlmostEqual(1.0, roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]))

    def test_perfect_inversion_is_zero(self):
        self.assertAlmostEqual(0.0, roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]))

    def test_all_ties_is_a_coin_flip(self):
        self.assertAlmostEqual(0.5, roc_auc([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]))

    def test_a_single_tie_across_the_boundary(self):
        """Three of four pairs ordered correctly, one tied, so 0.75 + 0.5 * 0.25"""
        self.assertAlmostEqual(0.875, roc_auc([0.1, 0.5, 0.5, 0.9], [0, 0, 1, 1]))

    def test_scale_and_shift_do_not_matter(self):
        scores = [0.1, 0.4, 0.35, 0.8]
        labels = [0, 0, 1, 1]
        self.assertAlmostEqual(roc_auc(scores, labels), roc_auc([100 * value + 7 for value in scores], labels))

    def test_one_class_is_undefined_rather_than_half(self):
        with self.assertRaises(MetricError):
            roc_auc([0.1, 0.2], [1, 1])

    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(MetricError):
            roc_auc([0.1, 0.2], [1])

class TestAccuracy(TestCase):
    def test_counts_the_side_of_the_threshold(self):
        self.assertAlmostEqual(1.0, accuracy([-1.0, -2.0, 1.0, 2.0], [0, 0, 1, 1]))
        self.assertAlmostEqual(0.5, accuracy([-1.0, -2.0, -1.0, -2.0], [0, 0, 1, 1]))

    def test_best_threshold_finds_the_separating_cut(self):
        threshold, best = best_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertAlmostEqual(1.0, best)
        self.assertTrue(0.2 < threshold < 0.8, threshold)

class TestCost(TestCase):
    def test_per_item_time_divides_through(self):
        self.assertAlmostEqual(5.0, Cost(seconds=1.0, items=200).ms_per_item)

    def test_measure_reports_after_the_block(self):
        with measure(items=4) as cost:
            sum(range(1000))
        self.assertEqual(4, cost[0].items)
        self.assertGreater(cost[0].seconds, 0.0)

    def test_measure_still_reports_when_the_block_raises(self):
        with self.assertRaises(ValueError), measure(items=1) as cost:
            raise ValueError("boom")
        self.assertEqual(1, cost[0].items)

class TestLogitDifference(TestCase):
    def test_it_subtracts_the_two_named_tokens(self):
        logits = torch.tensor([[1.0, 4.0, 0.0], [2.0, 0.0, 0.5]])
        self.assertEqual([3.0, 1.5], logit_difference(logits, [1, 0], [0, 2]).tolist())

    def test_each_row_gets_its_own_pair(self):
        """Every IOI prompt asks about different names, so the pair is per row"""
        logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
        self.assertEqual([5.0, 5.0], logit_difference(logits, [1, 0], [0, 1]).tolist())

    def test_a_shared_offset_cancels(self):
        logits = torch.tensor([[1.0, 4.0]])
        shifted = logits + 100.0
        self.assertEqual(logit_difference(logits, [1], [0]).tolist(), logit_difference(shifted, [1], [0]).tolist())

    def test_mismatched_ids_are_refused(self):
        with self.assertRaises(MetricError):
            logit_difference(torch.zeros(2, 3), [0], [0, 1])

    def test_it_needs_a_batch_of_logits(self):
        with self.assertRaises(MetricError):
            logit_difference(torch.zeros(3), [0], [1])

class TestRecovery(TestCase):
    def test_the_baselines_are_zero_and_one(self):
        self.assertAlmostEqual(0.0, recovery(patched=-1.0, clean=3.0, corrupted=-1.0))
        self.assertAlmostEqual(1.0, recovery(patched=3.0, clean=3.0, corrupted=-1.0))

    def test_halfway_is_a_half(self):
        self.assertAlmostEqual(0.5, recovery(patched=1.0, clean=3.0, corrupted=-1.0))

    def test_overshooting_and_undershooting_are_real_answers(self):
        self.assertGreater(recovery(patched=5.0, clean=3.0, corrupted=-1.0), 1.0)
        self.assertLess(recovery(patched=-3.0, clean=3.0, corrupted=-1.0), 0.0)

    def test_no_span_is_zero_rather_than_an_infinity(self):
        """A corruption that corrupted nothing cannot be divided by"""
        self.assertEqual(0.0, recovery(patched=2.0, clean=1.0, corrupted=1.0))
