import re
from types import SimpleNamespace
from unittest import TestCase

from src.core.ioi import (
    CORRUPTIONS,
    FRAMES,
    NAMES,
    IOIError,
    build_ioi,
    single_token_names,
)

"""
The IOI task is tested against a stub tokenizer rather than a checkpoint,
because everything that can go wrong with it is structural: a corruption that
does not corrupt, an order that is not balanced, a prompt that tokenizes to a
different length than its twin. None of those need a model to be wrong, and
all of them are invisible in the resulting numbers.

The stub splits the way GPT-2's BPE happens to split these frames -- a leading
space glued to each word, punctuation on its own -- so the landmark positions
the tests assert are the positions a real tokenizer produces. It also refuses
two-word names, which is the failure single_token_names exists to catch.
"""

class StubAdapter:
    """A tokenizer-shaped stand-in: one token per word, leading space included"""

    def __init__(self, multi_token=()):
        self.cfg = SimpleNamespace(id="stub", n_layers=4, n_heads=2)
        self.multi_token = set(multi_token)

    def tokens(self, prompt):
        return re.findall(r"\s*[\w']+|\s*[^\w\s]", prompt)

    def single_token(self, text):
        if text.strip() in self.multi_token or len(self.tokens(text)) != 1:
            raise ValueError(f"'{text}' is not a single token")
        return abs(hash(text)) % 50000

class TestBuild(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_it_alternates_the_two_name_orders(self):
        """A model that always says the first name has to score half, not all"""
        dataset = build_ioi(self.adapter, size=20, seed=0)
        self.assertEqual(dataset.balance, 0.5)

    def test_the_answer_is_the_name_that_is_not_repeated(self):
        dataset = build_ioi(self.adapter, size=8, seed=1)
        for example in dataset.examples:
            self.assertEqual(example.clean.count(example.subject), 2)
            self.assertEqual(example.clean.count(example.io), 1)

    def test_abc_corruption_replaces_only_the_repeated_name(self):
        dataset = build_ioi(self.adapter, size=8, seed=2, corruption="abc")
        for example in dataset.examples:
            self.assertIn(example.io, example.corrupted)
            self.assertEqual(example.corrupted.count(example.subject), 1)

    def test_swap_corruption_makes_the_other_name_correct(self):
        """Exchanging the roles has to move the repeated name, or nothing changed"""
        dataset = build_ioi(self.adapter, size=8, seed=3, corruption="swap")
        for example in dataset.examples:
            self.assertEqual(example.corrupted.count(example.io), 2)
            self.assertEqual(example.corrupted.count(example.subject), 1)

    def test_every_prompt_has_the_same_token_length(self):
        for corruption in CORRUPTIONS:
            with self.subTest(corruption=corruption):
                dataset = build_ioi(self.adapter, size=16, seed=4, corruption=corruption)
                lengths = {len(self.adapter.tokens(text)) for text in dataset.clean + dataset.corrupted}
                self.assertEqual(len(lengths), 1)

    def test_it_is_reproducible_from_its_seed(self):
        first = build_ioi(self.adapter, size=12, seed=7)
        second = build_ioi(self.adapter, size=12, seed=7)
        self.assertEqual(first.clean, second.clean)
        self.assertEqual(first.corrupted, second.corrupted)

    def test_every_frame_builds(self):
        for frame in range(len(FRAMES)):
            with self.subTest(frame=frame):
                self.assertEqual(len(build_ioi(self.adapter, size=4, seed=0, frame=frame)), 4)

class TestRejections(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_an_unknown_corruption_is_an_error(self):
        with self.assertRaises(IOIError) as caught:
            build_ioi(self.adapter, corruption="scramble")
        self.assertIn("scramble", str(caught.exception))

    def test_a_frame_that_does_not_exist_is_an_error(self):
        with self.assertRaises(IOIError):
            build_ioi(self.adapter, frame=len(FRAMES))

    def test_too_few_single_token_names_is_an_error(self):
        """Two names leaves nothing to corrupt with, and the message has to say so"""
        adapter = StubAdapter(multi_token=NAMES[2:])
        with self.assertRaises(IOIError) as caught:
            build_ioi(adapter, size=4)
        self.assertIn("three", str(caught.exception))

    def test_names_that_split_are_dropped_rather_than_used(self):
        adapter = StubAdapter(multi_token=("John", "Mary"))
        kept = single_token_names(adapter)
        self.assertNotIn("John", kept)
        self.assertNotIn("Mary", kept)
        self.assertIn("Tom", kept)

    def test_prompts_of_unequal_length_are_refused(self):
        """A name that is two tokens breaks the position alignment patching needs"""
        adapter = StubAdapter()
        original = adapter.single_token

        def lenient(text):
            try:
                return original(text)
            except ValueError:
                return 1

        adapter.single_token = lenient
        with self.assertRaises(IOIError) as caught:
            build_ioi(adapter, size=6, seed=0, names=["Al", "Bo", "Von Trapp"])
        self.assertIn("line up", str(caught.exception))

class TestLandmarks(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_the_three_mentions_and_the_end_are_found(self):
        dataset = build_ioi(self.adapter, size=2, seed=0)
        landmarks = dataset.landmarks(self.adapter)
        tokens = dataset.token_labels(self.adapter)
        example = dataset.examples[0]

        self.assertEqual(tokens[landmarks["IO"]].strip(), example.io)
        self.assertEqual(tokens[landmarks["S1"]].strip(), example.subject)
        self.assertEqual(tokens[landmarks["S2"]].strip(), example.subject)
        self.assertEqual(landmarks["END"], len(tokens) - 1)

    def test_the_repeated_mention_comes_after_both_names(self):
        dataset = build_ioi(self.adapter, size=2, seed=5)
        landmarks = dataset.landmarks(self.adapter)
        self.assertGreater(landmarks["S2"], landmarks["S1"])
        self.assertGreater(landmarks["S2"], landmarks["IO"])
        self.assertGreater(landmarks["END"], landmarks["S2"])

    def test_an_empty_dataset_has_no_landmarks(self):
        dataset = build_ioi(self.adapter, size=1, seed=0)
        dataset.examples = []
        with self.assertRaises(IOIError):
            dataset.landmarks(self.adapter)
