"""Guards on how translations are scored and when a difference counts.

Three things have to hold before any table in the study is readable: a scorer
refuses hypotheses that do not line up with their references (a silent
truncation scores the wrong sentences), the significance test reports both the
raw and the FDR-corrected verdict, and the survival frontier is the last share
before the first break rather than the luckiest intact seed anywhere.
"""

from unittest import TestCase

from src.methods.quality import (
    QualityError,
    agreement,
    bleu,
    bleu_signature,
    chrf,
    paired_significance,
    survival_frontier,
)

REFERENCES = [
    "The cat sat on the mat.",
    "It rained all night in the valley.",
    "She opened the window and looked out.",
    "The train was late again this morning.",
    "He forgot his keys on the kitchen table.",
    "The meeting ended without a decision.",
]
WORSE = [
    "The cat sat on a mat.",
    "It rained all the night in valley.",
    "She opened window and looked.",
    "Train was late this morning again.",
    "He forgot keys on the table.",
    "Meeting ended with no decision.",
]


class TestScorers(TestCase):
    def test_a_perfect_hypothesis_set_scores_full_marks(self):
        self.assertAlmostEqual(100.0, bleu(REFERENCES, REFERENCES), places=2)
        self.assertAlmostEqual(100.0, chrf(REFERENCES, REFERENCES), places=2)

    def test_a_worse_set_scores_lower(self):
        self.assertLess(bleu(WORSE, REFERENCES), bleu(REFERENCES, REFERENCES))
        self.assertLess(chrf(WORSE, REFERENCES), chrf(REFERENCES, REFERENCES))

    def test_mismatched_lengths_are_refused_rather_than_truncated(self):
        with self.assertRaises(QualityError):
            bleu(REFERENCES[:3], REFERENCES)
        with self.assertRaises(QualityError):
            chrf(REFERENCES[:3], REFERENCES)

    def test_the_signature_names_the_scorer_settings(self):
        """The string that makes a BLEU comparable with somebody else's"""
        signature = bleu_signature(WORSE, REFERENCES)
        self.assertTrue(signature.startswith("BLEU = "), signature)
        self.assertIn("nrefs:1", signature)
        self.assertIn("tok:13a", signature)


class TestPairedSignificance(TestCase):
    def test_every_system_gets_a_raw_and_a_corrected_verdict(self):
        outcome = paired_significance(REFERENCES, {"worse": WORSE, "same": list(REFERENCES)}, REFERENCES,
                                      resamples=40)
        self.assertEqual(2, len(outcome["components"]))
        self.assertEqual(len(REFERENCES), outcome["sentences"])
        for row in outcome["components"].values():
            self.assertIn("p", row)
            self.assertIn("q", row)
            self.assertIn("significant_fdr", row)
        self.assertAlmostEqual(0.0, outcome["components"]["same"]["dbleu"], places=2)
        self.assertGreater(outcome["components"]["worse"]["dbleu"], 0.0)

    def test_a_short_system_or_no_system_is_refused(self):
        with self.assertRaises(QualityError):
            paired_significance(REFERENCES, {"short": WORSE[:2]}, REFERENCES, resamples=10)
        with self.assertRaises(QualityError):
            paired_significance(REFERENCES, {}, REFERENCES, resamples=10)


class TestAgreement(TestCase):
    def test_the_same_ranking_agrees_and_the_inverse_does_not(self):
        first = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.assertTrue(agreement(first, first)["agrees"])
        # inverted is a perfect correlation of the wrong sign, which is a bug, not agreement
        inverted = agreement(first, list(reversed(first)))
        self.assertAlmostEqual(-1.0, inverted["spearman_rho"])
        self.assertFalse(inverted["agrees"])

    def test_too_few_items_is_not_a_test(self):
        with self.assertRaises(QualityError):
            agreement([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        with self.assertRaises(QualityError):
            agreement([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0])


class TestSurvivalFrontier(TestCase):
    def test_the_frontier_is_the_last_share_before_the_first_break(self):
        rows = [
            {"share": 0.01, "degeneracy": 0.0},
            {"share": 0.02, "degeneracy": 0.0},
            {"share": 0.05, "degeneracy": 0.1},
            {"share": 0.06, "degeneracy": 0.0},
        ]
        self.assertEqual((0.02, 0.05), survival_frontier(rows))

    def test_one_broken_seed_breaks_the_share(self):
        """A lucky seed at a share must not outrank a break at the same share"""
        rows = [
            {"share": 0.01, "degeneracy": 0.0, "seed": 0},
            {"share": 0.02, "degeneracy": 0.0, "seed": 0},
            {"share": 0.02, "degeneracy": 0.3, "seed": 1},
        ]
        self.assertEqual((0.01, 0.02), survival_frontier(rows))

    def test_nothing_broke_means_no_break_share(self):
        self.assertEqual((0.05, None), survival_frontier([{"share": 0.05, "degeneracy": 0.0}]))
        self.assertEqual((None, None), survival_frontier([]))
