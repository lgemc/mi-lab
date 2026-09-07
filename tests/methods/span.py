from unittest import TestCase

from src.core.readout import ReadoutError
from src.methods.circuits.techniques import TECHNIQUES, DiscoveryError, rank, require_technique, techniques_for
from src.methods.common.errors import CircuitError
from src.methods.common.span import baselines, behaviour

from ..stubs.fake import FakeAdapter, FakeTask, NoGradient, Sum

"""
The two references every causal number in `methods` is a fraction of, measured
against a model that is not a language model.

That is the whole test. `Baselines` carried `io` and `subject` -- two lists of
token ids -- into every downstream consumer, so nothing below `methods/common/`
could be exercised without a tokenizer. Here the adapter reads the *length* of
its inputs and nothing else, the score is a mean, and `span` and `recovery`
come out as the arithmetic they always were.

No checkpoint, so these run in the offline suite where the claim they check is
easiest to break.
"""

class TestBaselinesAreReadoutAgnostic(TestCase):
    def test_a_span_is_measured_with_no_tokenizer_touched(self):
        found = baselines(FakeAdapter(), FakeTask())
        # "clean-00" is 8 characters and "corrupt00" is 9, so the corruption moves it by one
        self.assertAlmostEqual(found.clean, 8.0)
        self.assertAlmostEqual(found.corrupted, 9.0)
        self.assertAlmostEqual(found.span, -1.0)

    def test_it_carries_the_readout_rather_than_two_lists_of_ids(self):
        found = baselines(FakeAdapter(), FakeTask())
        self.assertEqual("sum", found.readout.name)
        self.assertEqual("arbitrary", found.units)
        self.assertFalse(hasattr(found, "io"))

    def test_recovery_is_zero_at_corrupted_and_one_at_clean(self):
        found = baselines(FakeAdapter(), FakeTask())
        self.assertAlmostEqual(found.recovery(found.corrupted), 0.0)
        self.assertAlmostEqual(found.recovery(found.clean), 1.0)

    def test_a_corruption_that_corrupted_nothing_names_the_readout_that_did_not_move(self):
        task = FakeTask()
        task.corrupted = list(task.clean)
        with self.assertRaises(CircuitError) as caught:
            baselines(FakeAdapter(), task)
        self.assertIn("'sum'", str(caught.exception))
        self.assertIn("arbitrary", str(caught.exception))

    def test_a_score_that_cannot_say_what_it_measures_is_refused_before_the_pass(self):
        class Blank(Sum):
            definition = ""
        with self.assertRaises(ReadoutError):
            baselines(FakeAdapter(), FakeTask(score=Blank()))

class TestBehaviour(TestCase):
    def test_it_reports_the_score_and_the_units_it_is_in(self):
        clean = behaviour(FakeAdapter(), FakeTask())
        self.assertAlmostEqual(clean.score, 8.0)
        self.assertEqual("arbitrary", clean.units)
        self.assertEqual(4, clean.n)

    def test_accuracy_is_the_share_the_readout_scores_above_zero(self):
        """The one requirement on a Score's sign: more of the behaviour is a larger number"""
        self.assertAlmostEqual(behaviour(FakeAdapter(), FakeTask()).accuracy, 1.0)
        self.assertAlmostEqual(behaviour(FakeAdapter(scale=-1.0), FakeTask()).accuracy, 0.0)

class TestATechniqueRefusesAReadoutItCannotDifferentiate(TestCase):
    def test_the_refusal_lands_before_any_forward_pass(self):
        with self.assertRaises(DiscoveryError) as caught:
            rank("eap", FakeAdapter(), FakeTask(score=NoGradient()))
        message = str(caught.exception)
        self.assertIn("eap needs a gradient through the score", message)
        self.assertIn("'no-gradient'", message)
        # both halves of the message: what can be run, and what cannot
        self.assertIn("patching", message)
        self.assertIn("eap_ig", message)

    def test_a_differentiable_readout_gets_past_it(self):
        self.assertIs(require_technique("eap", FakeAdapter(), FakeTask()), TECHNIQUES["eap"])

    def test_every_technique_is_available_to_a_readout_with_a_gradient(self):
        self.assertEqual(sorted(TECHNIQUES), techniques_for(Sum()))
        gradient_free = techniques_for(NoGradient())
        self.assertNotIn("eap", gradient_free)
        self.assertIn("random", gradient_free)

    def test_an_unknown_technique_is_still_named_before_anything_else(self):
        with self.assertRaises(DiscoveryError) as caught:
            rank("telepathy", FakeAdapter(), FakeTask())
        self.assertIn("unknown technique", str(caught.exception))
