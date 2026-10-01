"""The frozen-dictionary diff's arithmetic on numbers chosen so the answer is known"""

from pathlib import Path
from unittest import TestCase

import torch

from src.domains.lm.analysis.features import SUMS, FeatureDiff
from src.domains.lm.data.sdft import SDFTDataError, results_name


def sums(res_pre, cent_pre, res_post, cent_post, dy=1.0, dy_miss=0.5, y_pre=10.0):
    row = {"res_pre": res_pre, "cent_pre": cent_pre, "res_post": res_post, "cent_post": cent_post,
           "dy": dy, "dy_miss": dy_miss, "y_pre": y_pre}
    return [row[name] for name in SUMS]


class TestPooledFit(TestCase):
    def test_fvu_pools_sums_rather_than_averaging_ratios(self):
        diff = FeatureDiff(text_set="t", layers=1, features=4)
        diff.add([sums(1.0, 10.0, 1.0, 10.0)], positions=10, half=0)     # ratio 0.1 on a long sequence
        diff.add([sums(1.0, 1.0, 1.0, 1.0)], positions=1, half=1)        # ratio 1.0 on a short one
        fit = diff.fit(resamples=50)
        self.assertAlmostEqual(fit["fvu_pre"]["per_layer"][0], 2.0 / 11.0)   # pooled, not (0.1 + 1) / 2

    def test_visible_change_is_one_minus_the_missed_share(self):
        diff = FeatureDiff(text_set="t", layers=1, features=4)
        diff.add([sums(1, 1, 1, 1, dy=4.0, dy_miss=1.0)], positions=5, half=0)
        self.assertAlmostEqual(diff.fit(resamples=10)["visible_change"]["per_layer"][0], 0.75)

    def test_the_interval_contains_the_point(self):
        diff = FeatureDiff(text_set="t", layers=1, features=4)
        for index in range(20):
            diff.add([sums(1 + index % 3, 10, 2, 10)], positions=5, half=index % 2)
        fit = diff.fit(resamples=200)["fvu_delta"]
        low, high = fit["per_layer_ci95"][0]
        self.assertLessEqual(low, fit["per_layer"][0])
        self.assertGreaterEqual(high, fit["per_layer"][0])


class TestFeatureReport(TestCase):
    def report(self, pre, post, halves=(50, 50)):
        diff = FeatureDiff(text_set="t", layers=1, features=len(pre[0]))
        diff.counts = torch.tensor([[[pre[0]], [pre[1]]], [[post[0]], [post[1]]]], dtype=torch.float32)
        diff.halves = list(halves)
        diff.positions = sum(halves)
        return diff.feature_report(floor=0.05)["layers"][0]

    def test_a_feature_that_quadruples_moves_across_and_not_within(self):
        layer = self.report(pre=[[5, 5, 0], [5, 5, 0]], post=[[20, 5, 0], [20, 5, 0]])
        self.assertEqual(layer["moved_paired"], 1)
        self.assertEqual(layer["moved_across_halves"], 1)
        self.assertEqual(layer["moved_within_halves"], 0)
        self.assertEqual(layer["top_shifted"][0]["feature"], 0)

    def test_sampling_noise_shows_up_on_both_sides_alike(self):
        # The same model twice, but the halves disagree: within and across must agree.
        layer = self.report(pre=[[20, 5, 0], [5, 5, 0]], post=[[20, 5, 0], [5, 5, 0]])
        self.assertEqual(layer["moved_paired"], 0)
        self.assertEqual(layer["moved_across_halves"], layer["moved_within_halves"])

    def test_jaccard_at_the_floor(self):
        layer = self.report(pre=[[5, 5, 0], [5, 5, 0]], post=[[5, 0, 5], [5, 0, 5]])
        self.assertAlmostEqual(layer["jaccard_paired"], 1 / 3)
        self.assertAlmostEqual(layer["jaccard_within_halves"], 1.0)
        self.assertAlmostEqual(layer["jaccard_across_halves"], 1 / 3)


class TestResultsName(TestCase):
    def test_a_hub_name_and_a_path_inside_the_repo(self):
        repo = Path("/repo")
        self.assertEqual(results_name("Qwen/Qwen3-0.6B", repo), "Qwen_Qwen3-0.6B")
        self.assertEqual(results_name("/repo/runs/seq/1-tooluse", repo), "runs_seq_1-tooluse")

    def test_a_path_outside_the_repo_is_refused(self):
        with self.assertRaises(SDFTDataError):
            results_name("/elsewhere/model", Path("/repo"))


class TestScore(TestCase):
    """The port of the SDFT repo's eval rules; parity with its saved flags was checked on 3,020 answers"""

    def test_science_reads_the_last_answer_tag(self):
        from src.domains.lm.data.sdft import score
        self.assertEqual(score("science", "<reasoning>x</reasoning>\n<answer>\nC\n</answer>", "C"), 1)
        self.assertEqual(score("science", "<answer>B</answer> then <answer>C</answer>", "B"), 0)

    def test_tool_use_scores_after_think_as_a_multiset_with_exact_inputs(self):
        from src.domains.lm.data.sdft import score
        golden = [{"Action": "search", "Action_Input": '{"q": "cats"}'}]
        good = '<think>Action: wrong</think>\nAction: search\nAction Input: {"q": "cats"}'
        self.assertEqual(score("tooluse", good, golden), 1)
        self.assertEqual(score("tooluse", 'Action: search\nAction Input: {"q": "dogs"}', golden), 0)
        self.assertEqual(score("tooluse", 'Action: search\nAction: search\nAction Input: {"q": "cats"}', golden), 0)
