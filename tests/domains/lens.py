from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from src.domains.lm.analysis.lens import (
    ItemScore,
    JacobianLens,
    logit_lens,
    matches,
    normalize,
    report,
    score_bank,
    score_item,
)

from ..stubs.model import shared_adapter

"""
The scorers are where a readout result is quietly made too good, so they are
tested harder than the lens itself: a matcher that accepts 'a' for 'apple'
turns any model into a mind reader, and a hit rate that counts items whose
answer already said the word is measuring the output, not the workspace.

The lens has one test worth running online, in tests/adapter-style: at the top
of the stack it must agree with the model's own logits, because there it *is*
the model's own logits. That one lives in TestLensAgreesAtTheTop and needs a
checkpoint.
"""

class TestNormalize(TestCase):
    def test_a_leading_space_marker_is_spelling_not_content(self):
        self.assertEqual("carnaval", normalize("Ġcarnaval"))
        self.assertEqual("carnaval", normalize(" carnaval"))
        self.assertEqual("carnaval", normalize("▁Carnaval"))

    def test_punctuation_around_a_token_is_dropped(self):
        self.assertEqual("iron", normalize(' "Iron".'))

class TestMatches(TestCase):
    def test_a_bpe_fragment_of_the_target_counts(self):
        self.assertTrue(matches("Feb", "February"))

    def test_a_token_swallowing_the_target_counts(self):
        self.assertTrue(matches("ironwork", "iron"))

    def test_a_short_prefix_is_a_coincidence_not_a_hit(self):
        self.assertFalse(matches("a", "apple"))
        self.assertFalse(matches("ca", "carnaval"))

    def test_a_different_word_is_not_a_hit(self):
        self.assertFalse(matches("copper", "iron"))

    def test_empty_matches_nothing(self):
        self.assertFalse(matches("", "iron"))
        self.assertFalse(matches(" ", "iron"))

class TestScoreItem(TestCase):
    ITEM = {"name": "fe", "prompt": "atomic number 26 is ", "intermediates": ["iron"]}

    def _readouts(self, *per_layer):
        from src.domains.lm.analysis.lens import LensReadout
        return [LensReadout(layer=index, tokens=list(tokens), scores=[0.5] * len(tokens))
                for index, tokens in enumerate(per_layer)]

    def test_the_first_hitting_layer_is_the_best_layer(self):
        score = score_item(self.ITEM, self._readouts(["the"], [" iron"], [" iron"]), "Fe", "multihop")
        self.assertEqual([1, 2], score.hit_layers)
        self.assertEqual(1, score.best_layer)
        self.assertTrue(score.hit)

    def test_no_layer_hitting_is_not_a_hit(self):
        score = score_item(self.ITEM, self._readouts(["the"], ["copper"]), "Fe", "multihop")
        self.assertFalse(score.hit)
        self.assertIsNone(score.best_layer)

    def test_an_answer_containing_the_intermediate_is_a_leak(self):
        score = score_item(self.ITEM, self._readouts([" iron"]), "iron, symbol Fe", "multihop")
        self.assertTrue(score.leaked)

    def test_an_answer_that_kept_the_word_to_itself_is_not(self):
        score = score_item(self.ITEM, self._readouts([" iron"]), "Fe", "multihop")
        self.assertFalse(score.leaked)

class TestReport(TestCase):
    def _score(self, hit_layers, leaked=False, answer="Fe"):
        return ItemScore(name="x", family="multihop", intermediates=["iron"], answer=answer,
                         hit_layers=list(hit_layers), best_layer=hit_layers[0] if hit_layers else None,
                         leaked=leaked)

    def test_leaked_items_are_kept_out_of_the_clean_rate(self):
        summary = report([self._score([3]), self._score([3], leaked=True, answer="iron"), self._score([])])
        self.assertEqual(0.6667, summary["hit_rate"])
        self.assertEqual(0.5, summary["clean_hit_rate"])   # one hit, one miss, the leak dropped
        self.assertEqual(0.3333, summary["leak_rate"])

    def test_the_layer_curve_counts_every_layer_that_hit(self):
        summary = report([self._score([2, 3]), self._score([3])])
        self.assertEqual({"2": 0.5, "3": 1.0}, summary["hit_rate_by_layer"])

    def test_an_empty_run_says_so_rather_than_dividing_by_zero(self):
        self.assertEqual({"items": 0}, report([]))

class TestLensAgreesAtTheTop(TestCase):
    """Needs GPT-2 small, like tests.adapter

    The lens is only believable if it reduces to the model at the top of the
    stack: layer n-1's residual, through the same final norm and the same
    unembedding, IS the model's next-token distribution. A lens that disagrees
    there is reading a different quantity at every other layer too.
    """

    PROMPT = "The Eiffel Tower is in the city of"

    @classmethod
    def setUpClass(cls):
        cls.adapter = shared_adapter()
        if cls.adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available")

    def test_the_last_layer_is_the_models_own_next_token(self):
        from src.domains.lm.analysis.lens import logit_lens
        top = self.adapter.cfg.n_layers - 1
        readout = logit_lens(self.adapter, [self.PROMPT], layers=[top], top_k=1)[0][0]
        expected = self.adapter.tokenizer.decode([int(self.adapter.logits([self.PROMPT])[0].argmax())])
        self.assertEqual(expected, readout.tokens[0])
        self.assertEqual(top, readout.layer)

    def test_every_layer_answers_with_a_distribution(self):
        from src.domains.lm.analysis.lens import logit_lens
        rows = logit_lens(self.adapter, [self.PROMPT], layers=[0, 6, 11], top_k=5)[0]
        self.assertEqual([0, 6, 11], [row.layer for row in rows])
        for row in rows:
            self.assertEqual(5, len(row.tokens))
            self.assertEqual(sorted(row.scores, reverse=True), row.scores)
            self.assertTrue(all(0.0 <= score <= 1.0 for score in row.scores))


class TestJacobianLens(TestCase):
    """The J-lens is the logit lens with a matrix in front, so it is tested against it"""

    PROMPTS = ("The Eiffel Tower is in the city of", "Two plus two is")

    @classmethod
    def setUpClass(cls):
        cls.adapter = shared_adapter()
        if cls.adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available")
        cls.width = cls.adapter.cfg.d_model

    def lens(self, matrix_for):
        layers = range(self.adapter.cfg.n_layers - 1)
        return JacobianLens({layer: matrix_for(layer) for layer in layers}, n_prompts=3)

    def test_an_identity_lens_is_the_logit_lens(self):
        """With J = I everywhere the transport is a no-op, and any difference is the plumbing"""
        lens = self.lens(lambda layer: torch.eye(self.width, dtype=torch.float16))
        plain = logit_lens(self.adapter, self.PROMPTS, top_k=5)
        through = logit_lens(self.adapter, self.PROMPTS, top_k=5, transport=lens.transport)
        for row_plain, row_through in zip(plain, through, strict=True):
            self.assertEqual([r.tokens for r in row_plain], [r.tokens for r in row_through])

    def test_the_transport_is_h_times_j_transposed(self):
        torch.manual_seed(0)
        matrix = torch.randn(4, 4)
        lens = JacobianLens({0: matrix})
        residual = torch.randn(2, 4)
        torch.testing.assert_close(residual @ matrix.T, lens.transport(residual, 0))

    def test_a_layer_without_a_matrix_passes_through(self):
        """The last layer's Jacobian onto itself is the identity, and the reference fits none"""
        lens = JacobianLens({0: torch.eye(4)})
        residual = torch.randn(2, 4)
        self.assertIs(residual, lens.transport(residual, 3))

    def test_a_scrambling_lens_changes_the_readout(self):
        """The check above would pass if transport were never called; this one would not"""
        generator = torch.Generator().manual_seed(0)
        lens = self.lens(lambda layer: torch.randn(self.width, self.width, generator=generator).half())
        plain = logit_lens(self.adapter, self.PROMPTS, layers=[3], top_k=5)
        through = logit_lens(self.adapter, self.PROMPTS, layers=[3], top_k=5, transport=lens.transport)
        self.assertNotEqual(plain[0][0].tokens, through[0][0].tokens)

    def test_the_reference_file_format_round_trips(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lens.pt"
            torch.save({"J": {0: torch.eye(4), 1: 2 * torch.eye(4)}, "n_prompts": 7,
                        "source_layers": [0, 1], "d_model": 4}, path)
            lens = JacobianLens.load(path)
        self.assertEqual(([0, 1], 7, 4), (lens.layers, lens.n_prompts, lens.d_model))

    def test_a_file_without_jacobians_is_refused(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-lens.pt"
            torch.save({"weights": torch.eye(4)}, path)
            with self.assertRaisesRegex(ValueError, "not a Jacobian lens"):
                JacobianLens.load(path)

    def test_a_lens_for_another_width_is_refused(self):
        with self.assertRaisesRegex(ValueError, "wide"):
            JacobianLens({0: torch.eye(self.width + 1)}).check(self.adapter)

    def test_reused_answers_are_not_generated_again(self):
        """Answers depend on the model and not on the readout, so a second method reuses them"""
        items = [{"name": "a", "prompt": self.PROMPTS[0], "intermediates": ["Paris"]}]
        original = self.adapter.generate

        def refuse(*args, **kwargs):
            raise AssertionError("generated although every answer was given")
        self.adapter.generate = refuse
        try:
            scores = score_bank(self.adapter, items, answers={"a": " Paris, France"})
        finally:
            self.adapter.generate = original
        self.assertTrue(scores[0].leaked)
