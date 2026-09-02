"""Guards on the shared protocol of the phase 1b knockout study.

Every phase1b script reads the same files by key, resolves the same band on
whatever model is loaded, stops the greedy search by the same saturation
rule and skips the same redundant candidates. What is checked here is that
shared part with no model and no corpus: a missing file is a refusal that
names the script to run, a scope is one of two words, and the greedy rules
behave on lists.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.core.config import ModelConfig
from src.data.translation import EvalSplit
from src.experiment import translation_study as study
from src.methods.cost import CostModel
from src.telemetry.results import ENV_ROOT


class TemporaryRoot(TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        previous = {key: os.environ.get(key) for key in (ENV_ROOT, study.ENV_EVAL_SENTENCES)}
        os.environ[ENV_ROOT] = str(self.root)
        os.environ.pop(study.ENV_EVAL_SENTENCES, None)

        def restore():
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.addCleanup(restore)


class TestArtifacts(TemporaryRoot):
    def test_every_key_lands_under_the_results_root(self):
        for key in study.ARTIFACTS:
            self.assertEqual(self.root, study.artifact(key).parent)
        self.assertEqual("phase1b-flops-model.json", study.artifact("cost").name)

    def test_an_unknown_key_is_refused_with_the_known_ones(self):
        with self.assertRaises(study.StudyError) as caught:
            study.artifact("weights2")
        self.assertIn("cost", str(caught.exception))

    def test_a_missing_cost_model_names_the_script_that_writes_it(self):
        with self.assertRaises(study.StudyError) as caught:
            study.cost_model()
        self.assertIn("phase1b_flops", str(caught.exception))

    def test_a_written_cost_model_reads_back(self):
        cost = CostModel(head_macs=10, mlp_macs=100, n_heads=4, n_layers=3, context=8)
        study.artifact("cost").write_text(json.dumps(cost.to_dict()))
        self.assertEqual(cost, study.cost_model())

    def test_the_candidate_is_read_from_where_greedy_wrote_it(self):
        with self.assertRaises(study.StudyError):
            study.candidate_components()
        study.artifact("candidate").write_text(json.dumps({"evaluations": {"candidate": {"components": ["mlp:9"]}}}))
        self.assertEqual(["mlp:9"], study.candidate_components())
        study.artifact("candidate").write_text(json.dumps({"evaluations": {}}))
        with self.assertRaises(study.StudyError):
            study.candidate_components()

    def test_the_ceiling_is_the_frontier_once_it_is_measured(self):
        self.assertEqual((study.PREREGISTERED_CEILING, "pre-registered ceiling"), study.frontier_ceiling())
        self.assertFalse(study.frontier_measured())
        study.artifact("frontier").write_text(json.dumps({"survival_frontier_share": 0.06}))
        self.assertEqual((0.06, "survival frontier"), study.frontier_ceiling())
        self.assertTrue(study.frontier_measured())

    def test_the_eval_size_comes_from_the_environment(self):
        self.assertEqual(study.DEFAULT_EVAL_SENTENCES, study.eval_sentences())
        os.environ[study.ENV_EVAL_SENTENCES] = "50"
        self.assertEqual(50, study.eval_sentences())
        self.assertEqual(25, study.head_sentences())


class TestScope(TestCase):
    def setUp(self):
        self.cfg = ModelConfig(id="depth-12", backend="transformers", hf_name="none/tiny", n_layers=12, n_heads=2)

    def test_the_candidate_scope_is_the_band_and_the_model_scope_is_everything(self):
        self.assertEqual([9, 10, 11], study.scope_layers(self.cfg, "candidate"))
        self.assertEqual(list(range(12)), study.scope_layers(self.cfg, "model"))

    def test_another_scope_is_refused(self):
        with self.assertRaises(study.StudyError):
            study.scope_layers(self.cfg, "band")

    def test_the_pool_is_atomic_over_the_scope(self):
        pool = study.pool(self.cfg, "candidate")
        self.assertIn("mlp:9", pool)
        self.assertIn("head:11:1", pool)
        self.assertNotIn("heads:11", pool)

    def test_combos_are_the_three_whole_band_sets(self):
        sets = study.combos([10, 11])
        self.assertEqual(["heads:10", "heads:11"], sets["all_candidate_heads"])
        self.assertEqual(len(sets["all_candidate_mlps"]) + 2, len(sets["full_candidate_set"]))


class TestGreedy(TestCase):
    def test_saturation_needs_the_last_runs_all_below_the_margin(self):
        self.assertFalse(study.saturated([0.1, 0.1]))
        self.assertTrue(study.saturated([2.0, 0.1, 0.2, 0.1]))
        self.assertFalse(study.saturated([0.1, 0.1, 1.0]))

    def test_the_next_candidate_skips_chosen_and_redundant_components(self):
        ranking = ["mlp:9", "head:9:1", "heads:9", "mlp:10"]
        self.assertEqual("mlp:10", study.next_candidate(ranking, ["mlp:9", "head:9:1"]))
        self.assertIsNone(study.next_candidate(["mlp:9"], ["mlp:9"]))

    def test_describe_truncates_with_the_count(self):
        self.assertEqual("mlp:1, mlp:2", study.describe(["mlp:1", "mlp:2"]))
        self.assertTrue(study.describe([f"mlp:{i}" for i in range(9)]).endswith("(9 total)"))

    def test_degeneracy_is_recomputed_from_hypotheses_when_they_are_kept(self):
        self.assertEqual(0.5, study.rerun_degeneracy({"hypotheses": ["", "a fine sentence here"]}))
        self.assertEqual(0.25, study.rerun_degeneracy({"degeneracy": 0.25}))


class TestCorpus(TestCase):
    def test_a_part_slices_prompts_and_references_together(self):
        pairs = tuple((f"es {i}", f"en {i}") for i in range(6))
        corpus = study.Corpus(pairs=pairs, split=EvalSplit(shots=pairs[-2:], pairs=pairs[:4]))
        prompts, references = corpus.part(slice(1, 3))
        self.assertEqual(["en 1", "en 2"], references)
        self.assertEqual(2, len(prompts))
        self.assertIn("es 2", prompts[1])
        self.assertEqual(4, len(corpus))
        self.assertEqual(4, len(corpus.counterfactual))
