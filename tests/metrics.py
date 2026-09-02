from unittest import TestCase

import torch

from src.core.metrics import (
    Cost,
    MetricError,
    accuracy,
    benjamini_hochberg,
    best_threshold,
    degeneracy,
    jaccard,
    logit_difference,
    measure,
    recovery,
    roc_auc,
    spearman,
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

class TestSpearman(TestCase):
    def test_the_same_order_correlates_perfectly_whatever_the_scale(self):
        """Two techniques report in different units; only the order is comparable"""
        self.assertAlmostEqual(1.0, spearman([1.0, 2.0, 3.0], [10.0, 200.0, 3000.0]))

    def test_the_opposite_order_is_minus_one(self):
        self.assertAlmostEqual(-1.0, spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]))

    def test_ties_share_a_rank(self):
        """A technique that scores half the heads at zero states no order between them"""
        self.assertAlmostEqual(1.0, spearman([1.0, 1.0, 2.0], [5.0, 5.0, 9.0]))

    def test_a_constant_scoring_has_no_order_to_correlate(self):
        with self.assertRaises(MetricError):
            spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_two_scorings_have_to_rank_the_same_things(self):
        with self.assertRaises(MetricError):
            spearman([1.0, 2.0], [1.0, 2.0, 3.0])

class TestJaccard(TestCase):
    def test_the_same_set_overlaps_completely(self):
        self.assertEqual(1.0, jaccard({(9, 9), (10, 7)}, {(10, 7), (9, 9)}))

    def test_disjoint_sets_overlap_at_zero(self):
        self.assertEqual(0.0, jaccard({(0, 0)}, {(1, 1)}))

    def test_it_is_the_union_and_not_the_smaller_set(self):
        """Otherwise a circuit of two heads inside one of eighty would score 1.0"""
        self.assertAlmostEqual(0.25, jaccard({1, 2}, {1, 2, 3, 4, 5, 6, 7, 8}))

    def test_two_empty_sets_are_not_a_perfect_overlap(self):
        with self.assertRaises(MetricError):
            jaccard([], [])

class TestDegeneracy(TestCase):
    """The machine's hand raised when a generation stops being language"""

    def test_healthy_sentences_score_zero(self):
        self.assertEqual(0.0, degeneracy(["The cat sat on the mat today.", "It rained all night in the valley."]))
        self.assertEqual(0.0, degeneracy([]))

    def test_an_empty_generation_is_the_most_degenerate_outcome(self):
        """An early version fell through the too-short guard and scored silence as healthy"""
        self.assertEqual(0.5, degeneracy(["", "It rained all night in the valley."]))

    def test_punctuation_without_letters_is_not_a_translation_at_any_length(self):
        self.assertEqual(1.0, degeneracy(["...", "(   (   ("]))

    def test_a_repeated_token_is_collapse(self):
        self.assertEqual(1.0, degeneracy(["the the the the the the"]))
        self.assertEqual(1.0, degeneracy(["yes no yes yes yes yes yes yes"]))

    def test_the_same_short_answer_across_the_corpus_is_collapse(self):
        """Three tokens or fewer never reach the repetition rules; the signal is across hypotheses"""
        clean = [f"A different sentence number {i} here." for i in range(20)]
        self.assertEqual(0.0, degeneracy([*clean, "the", "the"]))
        self.assertAlmostEqual(3 / 23, degeneracy([*clean, "the", "the", "the"]), places=3)

class TestBenjaminiHochberg(TestCase):
    def test_q_values_are_monotone_in_p_and_never_below_it(self):
        raw = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
        q = benjamini_hochberg(raw)
        # b and c tie at .04: the running minimum from the bottom is what makes q monotone
        self.assertLessEqual(q["a"], q["b"])
        self.assertLessEqual(q["b"], q["c"])
        self.assertLessEqual(q["c"], q["d"])
        for name, p in raw.items():
            self.assertGreaterEqual(q[name], p)
        self.assertEqual(0.5, q["d"])

    def test_the_smallest_p_is_multiplied_by_the_number_of_tests(self):
        """One row at .01 among ten screened is a .1 finding"""
        raw = {f"c{i}": 0.9 for i in range(9)}
        raw["hit"] = 0.01
        self.assertAlmostEqual(0.1, benjamini_hochberg(raw)["hit"], places=5)

    def test_nothing_in_is_nothing_out(self):
        self.assertEqual({}, benjamini_hochberg({}))
