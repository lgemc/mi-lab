import json
import tempfile
from pathlib import Path
from unittest import TestCase

from src.core.dataset import _SUBJECTS, DatasetError, LabeledPrompts, load_jsonl, synthetic

"""
The dataset object is small enough that the only things worth testing are the
ones that quietly ruin a result: a split that leaks, a split that leaves a
class empty, and labels that are not what they claim to be.
"""

class TestLabeledPrompts(TestCase):
    def test_mismatched_lengths_are_refused(self):
        with self.assertRaises(DatasetError):
            LabeledPrompts(texts=["a", "b"], labels=[1])

    def test_labels_must_be_binary(self):
        with self.assertRaises(DatasetError):
            LabeledPrompts(texts=["a"], labels=[2])

    def test_balance_reports_the_positive_rate(self):
        data = LabeledPrompts(texts=list("abcd"), labels=[1, 1, 1, 0])
        self.assertEqual(0.75, data.balance)

class TestSplit(TestCase):
    def setUp(self):
        self.data = synthetic(40, seed=1)

    def test_split_keeps_every_example_exactly_once(self):
        train, test = self.data.split(0.25, seed=0)
        self.assertEqual(len(self.data), len(train) + len(test))
        self.assertEqual(sorted(self.data.texts), sorted(train.texts + test.texts))

    def test_both_classes_survive_on_both_sides(self):
        train, test = self.data.split(0.25, seed=0)
        for part in (train, test):
            self.assertGreater(part.positives, 0)
            self.assertLess(part.positives, len(part))

    def test_the_same_seed_gives_the_same_split(self):
        self.assertEqual(self.data.split(0.3, seed=7)[1].texts, self.data.split(0.3, seed=7)[1].texts)

    def test_a_nonsense_fraction_is_refused(self):
        for fraction in (0.0, 1.0, -0.5):
            with self.subTest(fraction=fraction), self.assertRaises(DatasetError):
                self.data.split(fraction)

class TestSynthetic(TestCase):
    def test_sentences_are_distinct_so_a_split_cannot_leak(self):
        """The bug this test exists for: sampling templates with replacement

        Duplicated sentences put the same example in train and test, and the
        reported AUC then measures memorization rather than generalization.
        """
        data = synthetic(200, seed=0)
        self.assertEqual(len(data), len(set(data.texts)))
        train, test = data.split(0.3, seed=0)
        self.assertEqual(set(), set(train.texts) & set(test.texts))

    def test_it_is_balanced(self):
        self.assertEqual(0.5, synthetic(100, seed=3).balance)

    def test_both_classes_use_the_same_subjects(self):
        """Otherwise a probe can separate the classes on topic alone"""
        data = synthetic(280, seed=0)
        subjects = {label: set() for label in (0, 1)}
        for text, label in zip(data.texts, data.labels):
            found = [subject for subject in _SUBJECTS if subject in text]
            self.assertEqual(1, len(found), f"expected exactly one subject in {text!r}, found {found}")
            subjects[label].add(found[0])
        self.assertEqual(subjects[0], subjects[1])
        self.assertEqual(set(_SUBJECTS), subjects[0])

    def test_asking_for_more_than_exists_says_the_cap(self):
        with self.assertRaises(DatasetError) as caught:
            synthetic(1000)
        self.assertIn("280", str(caught.exception))

class TestLoadJsonl(TestCase):
    def _write(self, rows) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.write("\n".join(json.dumps(row) for row in rows))
        handle.close()
        return handle.name

    def test_reads_texts_and_labels(self):
        path = self._write([{"text": "good", "label": 1}, {"text": "bad", "label": 0}])
        data = load_jsonl(path)
        self.assertEqual(["good", "bad"], data.texts)
        self.assertEqual([1, 0], data.labels)

    def test_field_names_are_configurable(self):
        path = self._write([{"prompt": "good", "y": 1}, {"prompt": "bad", "y": 0}])
        data = load_jsonl(path, text_field="prompt", label_field="y")
        self.assertEqual(["good", "bad"], data.texts)

    def test_a_missing_field_names_the_line(self):
        path = self._write([{"text": "good", "label": 1}, {"text": "bad"}])
        with self.assertRaises(DatasetError) as caught:
            load_jsonl(path)
        self.assertIn(":2", str(caught.exception))

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(DatasetError):
            load_jsonl(str(Path(tempfile.gettempdir()) / "definitely-not-here.jsonl"))
