from unittest import TestCase

import torch

from src.core.readout import (
    DirectScore,
    ReadoutError,
    Score,
    mean_score,
    require_direct,
    require_readout,
    require_score,
)
from src.domains.lm.readout import LogitDifference

from ..stubs.fake import FakeAdapter, NoGradient, Sum

"""
The third thing `methods/` consumes, and the refusals that keep it honest.

No checkpoint here on purpose. The whole claim of the measurement contract is
that a score is an object with a definition rather than a pair of token ids
threaded through four call sites, and an object with a definition can be checked
with no weights at all.
"""

class TestTheProtocol(TestCase):
    def test_a_well_formed_score_satisfies_it_structurally(self):
        self.assertIsInstance(Sum(), Score)
        self.assertIsInstance(LogitDifference([1], [2]), Score)

    def test_a_score_that_cannot_say_what_it_measures_is_refused(self):
        """An empty units is an unlabelled axis rather than an error, so it is an error here"""
        class Blank(Sum):
            units = ""
        with self.assertRaises(ReadoutError) as caught:
            require_score(Blank())
        self.assertIn("units", str(caught.exception))

    def test_something_that_is_not_a_score_at_all_names_what_is_missing(self):
        with self.assertRaises(ReadoutError) as caught:
            require_score(object())
        self.assertIn("definition", str(caught.exception))

class TestTheGradientMarker(TestCase):
    def test_a_readout_is_decided_by_the_field_and_not_by_the_protocol(self):
        """A Protocol with no members of its own is satisfied by everything

        Which is why `differentiable` is data the score declares rather than a
        base class: isinstance against an empty marker would pass for BLEU and
        the refusal would never fire.
        """
        from src.core.readout import Readout

        differentiable, flat = Sum(), NoGradient()
        # both satisfy the marker structurally, which is exactly the trap
        self.assertIsInstance(differentiable, Readout)
        self.assertIsInstance(flat, Readout)
        self.assertIs(require_readout(differentiable), differentiable)
        with self.assertRaises(ReadoutError) as caught:
            require_readout(NoGradient(), by="eap")
        self.assertIn("eap", str(caught.exception))
        self.assertIn("gradient through the score", str(caught.exception))

class TestTheDirectForm(TestCase):
    def test_a_score_with_no_direct_form_is_named_rather_than_missing_an_attribute(self):
        with self.assertRaises(ReadoutError) as caught:
            require_direct(Sum(), by="direct attribution")
        self.assertIn("direct attribution", str(caught.exception))
        self.assertIn("no direct form", str(caught.exception))

    def test_the_logit_difference_has_one(self):
        self.assertIsInstance(LogitDifference([1], [2]), DirectScore)

class TestMeanScore(TestCase):
    def test_it_is_the_mean_of_the_score_over_what_the_model_produced(self):
        adapter, score = FakeAdapter(), Sum()
        inputs = ["ab", "abcd"]
        self.assertAlmostEqual(mean_score(adapter, inputs, score), 3.0)
        self.assertEqual([2.0, 4.0], score(adapter.outputs(inputs)).tolist())

class TestLogitDifference(TestCase):
    def test_it_is_the_difference_of_the_two_logits_it_names(self):
        logits = torch.tensor([[0.0, 1.0, 4.0], [2.0, 0.0, 0.5]])
        self.assertEqual([3.0, 1.5], LogitDifference([2, 0], [1, 2])(logits).tolist())

    def test_it_keeps_the_graph_so_a_gradient_can_be_taken_through_it(self):
        """`differentiable = True` is a claim, and this is the thing that checks it

        A readout that says it is differentiable and detaches produces a
        gradient that looks fine and answers a different question, and only the
        finite-difference receipt in tests/methods/discovery.py would say so.
        """
        logits = torch.tensor([[0.0, 1.0, 4.0]], requires_grad=True)
        score = LogitDifference([2], [1])(logits).sum()
        self.assertIsNotNone(score.grad_fn)
        score.backward()
        self.assertEqual([[0.0, -1.0, 1.0]], logits.grad.tolist())

    def test_select_narrows_it_to_a_chunks_own_rows(self):
        """A score built for the whole batch, scored on a chunk, scores the wrong answers"""
        score = LogitDifference([1, 2, 3], [4, 5, 6], model="fake")
        narrowed = score.select(slice(1, 3))
        self.assertEqual([2, 3], narrowed.positive)
        self.assertEqual([5, 6], narrowed.negative)
        self.assertEqual("fake", narrowed.model)

    def test_the_two_id_lists_have_to_index_the_same_batch(self):
        with self.assertRaises(ReadoutError):
            LogitDifference([1, 2], [3])

    def test_it_refuses_a_model_it_was_not_built_against(self):
        """The ids are one checkpoint's opinion, so holding it past a swap is Means.check's bug"""
        adapter = FakeAdapter()
        LogitDifference([1], [2], model="fake").check(adapter)
        with self.assertRaises(ReadoutError) as caught:
            LogitDifference([1], [2], model="gpt2-small").check(adapter)
        self.assertIn("gpt2-small", str(caught.exception))
