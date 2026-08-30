import re
from types import SimpleNamespace
from unittest import TestCase

from src.data.ioi import build_ioi
from src.data.tasks import (
    TASKS,
    TaskError,
    TaskExample,
    TemplateTask,
    build_task,
    require_alignment,
    single_tokens,
    task_names,
)

"""
The task registry is tested against a stub tokenizer, for the same reason IOI
is: everything that can go wrong with a task is structural. A corruption that
changes two things at once, an answer that is also the distractor, prompts
whose twins tokenize to different lengths -- none of those need a checkpoint
to be wrong, and every one of them is invisible in the numbers that come out
the far end.

The stub splits one token per word with the leading space attached, which is
close enough to a BPE for the alignment checks to mean what they mean. What it
cannot check is whether GPT-2 actually keeps ' windows' whole; that is what
the pools are filtered against a real tokenizer for, and what tests/discovery
runs on a checkpoint.
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

class TestRegistry(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_every_registered_task_builds(self):
        for name in task_names():
            with self.subTest(task=name):
                self.assertEqual(len(build_task(name, self.adapter, size=6, seed=0)), 6)

    def test_every_task_is_reproducible_from_its_seed(self):
        for name in task_names():
            with self.subTest(task=name):
                first = build_task(name, self.adapter, size=6, seed=3)
                second = build_task(name, self.adapter, size=6, seed=3)
                self.assertEqual(first.clean, second.clean)
                self.assertEqual(first.corrupted, second.corrupted)

    def test_every_task_lines_up_position_for_position(self):
        """Patching reads position i of one run into position i of another"""
        for name in task_names():
            with self.subTest(task=name):
                task = build_task(name, self.adapter, size=8, seed=1)
                lengths = {len(self.adapter.tokens(text)) for text in task.clean + task.corrupted}
                self.assertEqual(len(lengths), 1, f"'{name}' produced prompts of {sorted(lengths)} lengths")

    def test_the_corruption_actually_corrupts(self):
        for name in task_names():
            with self.subTest(task=name):
                task = build_task(name, self.adapter, size=8, seed=1)
                for clean, corrupted in zip(task.clean, task.corrupted, strict=True):
                    self.assertNotEqual(clean, corrupted)

    def test_the_answer_is_never_the_distractor(self):
        for name in task_names():
            with self.subTest(task=name):
                task = build_task(name, self.adapter, size=8, seed=1)
                answer, distractor = task.answers(self.adapter)
                self.assertTrue(all(left != right for left, right in zip(answer, distractor, strict=True)))

    def test_every_task_carries_a_description(self):
        for name in task_names():
            self.assertTrue(TASKS[name].description.strip(), f"'{name}' is registered without a description")

    def test_an_unknown_task_names_the_ones_that_exist(self):
        with self.assertRaises(TaskError) as raised:
            build_task("sentiment", self.adapter)
        self.assertIn("ioi", str(raised.exception))

    def test_a_task_of_no_examples_is_refused(self):
        with self.assertRaises(TaskError):
            build_task("ioi", self.adapter, size=0)

class TestTemplates(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_greater_than_puts_the_answer_above_the_start_and_the_distractor_below(self):
        task = build_task("greater_than", self.adapter, size=12, seed=0)
        for example in task.examples:
            start = int(example.clean.split("year 17")[1][:2])
            self.assertGreater(int(example.answer), start)
            self.assertLess(int(example.distractor), start)

    def test_greater_than_corrupts_only_the_start_year(self):
        """The noun is drawn once per example: a pair differing in two things measures neither"""
        task = build_task("greater_than", self.adapter, size=8, seed=0)
        for example in task.examples:
            self.assertEqual(
                example.clean.split(" lasted")[0], example.corrupted.split(" lasted")[0]
            )
            self.assertIn("year 1701 to", example.corrupted)

    def test_induction_repeats_the_first_word_and_the_corruption_does_not(self):
        task = build_task("induction", self.adapter, size=8, seed=0)
        for example in task.examples:
            words = example.clean.removeprefix("Words:").split()
            self.assertEqual(words[0], words[-1])
            self.assertEqual(f" {words[1]}", example.answer)
            corrupted = example.corrupted.removeprefix("Words:").split()
            self.assertNotIn(corrupted[-1], corrupted[:-1])

    def test_agreement_alternates_the_two_numbers(self):
        """A task all in one number measures the number, not the agreement"""
        task = build_task("agreement", self.adapter, size=10, seed=0)
        self.assertEqual(task.variants, {"plural": 5, "singular": 5})

    def test_agreement_puts_the_attractor_in_the_opposite_number(self):
        task = build_task("agreement", self.adapter, size=8, seed=0)
        for example in task.examples:
            self.assertNotEqual(example.answer, example.distractor)
            # the corruption flips the subject, so the clean answer becomes the wrong one
            self.assertNotEqual(example.clean, example.corrupted)

class TestSubsets(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_a_subset_keeps_the_frame_and_cuts_the_examples(self):
        task = build_task("induction", self.adapter, size=8, seed=0)
        one = task.subset([3])
        self.assertEqual(len(one), 1)
        self.assertEqual(one.clean, [task.clean[3]])
        self.assertEqual(one.frame, task.frame)

    def test_an_ioi_subset_keeps_its_corruption(self):
        dataset = build_ioi(self.adapter, size=8, seed=0, corruption="swap")
        one = dataset.subset([0, 1])
        self.assertEqual(one.corruption, "swap")
        self.assertEqual(one.clean, dataset.clean[:2])

    def test_an_index_the_task_does_not_have_is_an_error(self):
        task = build_task("ioi", self.adapter, size=4, seed=0)
        with self.assertRaises(ValueError):
            task.subset([9])

class TestAlignment(TestCase):
    def setUp(self):
        self.adapter = StubAdapter()

    def test_prompts_of_unequal_length_are_refused(self):
        task = TemplateTask(
            examples=[
                TaskExample(clean="one two", corrupted="one three", answer=" a", distractor=" b"),
                TaskExample(clean="one two three", corrupted="one two four", answer=" a", distractor=" b"),
            ],
            name="ragged",
        )
        with self.assertRaises(TaskError) as raised:
            require_alignment(self.adapter, task)
        self.assertIn("line up position for position", str(raised.exception))

    def test_a_pool_is_filtered_against_the_tokenizer_in_hand(self):
        adapter = StubAdapter(multi_token={"lantern"})
        kept = single_tokens(adapter, (" apple", " lantern"))
        self.assertEqual(kept, [" apple"])

    def test_a_pool_too_small_to_fill_the_frame_is_an_error(self):
        adapter = StubAdapter(multi_token={f"{value:02d}" for value in range(1, 99)})
        with self.assertRaises(TaskError) as raised:
            build_task("greater_than", adapter, size=4)
        self.assertIn("single-token", str(raised.exception))
