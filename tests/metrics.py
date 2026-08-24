from unittest import TestCase

from src.core.metrics import Cost, MetricError, accuracy, best_threshold, measure, roc_auc

"""
AUC is the number every later result is quoted in, so it is worth pinning
against hand-computable cases: perfect separation, perfect inversion, and the
all-ties case where the answer must be exactly a coin flip.
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
        with self.assertRaises(ValueError):
            with measure(items=1) as cost:
                raise ValueError("boom")
        self.assertEqual(1, cost[0].items)
