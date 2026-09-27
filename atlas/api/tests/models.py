"""The toy models do what each lesson says they do

A lesson is a claim about its model. These pin the claims, so a change to a
weight that quietly breaks a sentence on a page fails here instead.
"""

import unittest

import numpy as np

from atlas import crosscoder, ioi, lens, mlp, probing, sae, superposition, transcoder


class TestMlp(unittest.TestCase):
    def test_it_computes_its_boolean_function(self):
        for row in mlp.truth_table():
            self.assertEqual(row["prob"] > 0.5, bool(row["target"]), row)

    def test_denoising_and_noising_disagree_about_the_redundant_neuron(self):
        rows = {r["node"]: r for r in mlp.patching([1, 1, 0], [1, 0, 0], "prob")["rows"]}
        self.assertGreater(rows["n1"]["recovery"], 0.9)  # sufficient
        self.assertLess(rows["n1"]["damage"], 0.1)  # but not necessary: n2 covers for it

    def test_attribution_patching_fails_on_the_saturated_sigmoid_and_ig_does_not(self):
        rows = {r["node"]: r for r in mlp.patching([1, 1, 0], [1, 0, 0], "prob")["rows"]}
        self.assertLess(rows["n1"]["attribution"], 0.5 * rows["n1"]["true_effect"])
        self.assertAlmostEqual(rows["n1"]["integrated"], rows["n1"]["true_effect"], delta=0.05)


class TestIoi(unittest.TestCase):
    def test_the_circuit_answers_with_the_indirect_object(self):
        for io, s, t in ioi.dataset():
            v = ioi.forward_view(io, s, t, {})
            self.assertGreater(v["logit_diff"], 1.5, (io, s, t))

    def test_abc_corruption_removes_the_preference(self):
        self.assertLess(abs(ioi.forward_view("Mary", "John", "ABBA", {}, "abc")["logit_diff"]), 0.2)

    def test_direct_logit_attribution_is_exact(self):
        v = ioi.forward_view("Alice", "Tom", "BABA", {})
        self.assertAlmostEqual(sum(v["dla"].values()), v["logit_diff"], places=9)

    def test_the_previous_token_head_is_irrelevant_and_the_name_mover_is_not(self):
        p = ioi.patch_heads("Mary", "John", "ABBA", "abc", "noise")["per_head"]
        self.assertAlmostEqual(p["L0.0"], 0.0, places=6)
        self.assertGreater(p["L2.0"], 1.0)

    def test_acdc_keeps_the_circuit_and_eap_misses_the_saturated_query_edge(self):
        e = ioi.edges("Mary", "John", "ABBA", "abc", 0.1)
        kept = {tuple(k) for k in e["acdc_kept"]}
        self.assertIn(("L1.0", "L2.0", "q"), kept)
        self.assertIn(("L0.1", "L1.0", "k"), kept)
        self.assertNotIn(("L0.0", "logits", "resid"), kept)
        row = next(r for r in e["edges"] if (r["sender"], r["receiver"], r["channel"]) == ("L1.0", "L2.0", "q"))
        self.assertLess(abs(row["eap"]), 0.1 * abs(row["patch"]))
        self.assertAlmostEqual(row["eap_ig"], row["patch"], delta=0.3 * abs(row["patch"]))


class TestLens(unittest.TestCase):
    def test_logit_lens_is_blind_to_the_workspace_and_the_j_lens_is_not(self):
        r = lens.probe("Curie")
        mid = r["layers"][2]
        self.assertEqual(mid["top"]["logit"][0]["token"], "Curie")
        self.assertEqual(mid["top"]["jlens"][0]["token"], "Poland")
        self.assertEqual(r["answer"][0]["token"], "Warsaw")

    def test_editing_the_workspace_changes_the_answer(self):
        self.assertEqual(lens.probe("Curie", "Japan", 2)["answer"][0]["token"], "Tokyo")

    def test_the_last_layer_lenses_agree_with_the_model(self):
        top = lens.probe("Kafka")["layers"][-1]["top"]
        self.assertEqual({v[0]["token"] for v in top.values()}, {"Prague"})


class TestDictionaries(unittest.TestCase):
    def test_superposition_appears_with_sparsity(self):
        self.assertEqual(superposition.train(5, 2, 0.0)["features_represented"], 2)
        self.assertEqual(superposition.train(5, 2, 0.9)["features_represented"], 5)

    def test_the_sae_finds_the_true_features(self):
        r = sae.train(5, 5, 0.15, 0.3, "relu", 1)
        self.assertGreater(min(r["best_match"]), 0.95)
        r = sae.train(5, 5, 0.15, 0.0, "topk", 1)
        self.assertGreater(min(r["best_match"]), 0.95)

    def test_the_transcoder_imitates_the_mlp_with_sparse_latents(self):
        r = transcoder.train(8, 0.1)
        self.assertLess(r["fvu"], 0.05)
        self.assertLess(r["l0"], 2.0)
        g = transcoder.attribution_graph([0, 0, 1, 0, 0])
        self.assertEqual(len(g["latents"]), 1)

    def test_the_crosscoder_separates_exclusive_features(self):
        r = crosscoder.train("l1")
        alive = [x for x in r["latents"] if x["alive"] and x["match_cos"] > 0.95]
        for x in alive:
            if x["match_kind"] == "chat":
                self.assertGreater(x["rel"], 0.9)
            if x["match_kind"] == "base":
                self.assertLess(x["rel"], 0.1)

    def test_steering_along_the_spurious_direction_does_nothing(self):
        r = probing.study()
        by = {x["name"]: x for x in r["steering"]}
        self.assertAlmostEqual(by["spurious direction"]["effect"], 0.0, places=6)
        self.assertGreater(by["true (used) direction"]["effect"], by["probe"]["effect"])
        self.assertGreater(r["single_direction_accuracy"]["spurious"], r["single_direction_accuracy"]["used"])


class TestNoNans(unittest.TestCase):
    def test_trainers_return_finite_numbers(self):
        for out in (sae.train(3, 8, 0.3, 1.0, "relu", 1, steps=300), superposition.train(8, 2, 0.99, 0.5, steps=300)):
            flat = np.array([v for k, v in out.items() if isinstance(v, float)])
            self.assertTrue(np.all(np.isfinite(flat)))


if __name__ == "__main__":
    unittest.main()
