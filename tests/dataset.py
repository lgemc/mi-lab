import json
import tempfile
from pathlib import Path
from unittest import TestCase

from src.data.dataset import _SUBJECTS, DatasetError, LabeledPrompts, load_csv, load_jsonl, save_jsonl, synthetic
from src.data.prompts import dumps, parse

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
        for text, label in zip(data.texts, data.labels, strict=True):
            found = [subject for subject in _SUBJECTS if subject in text]
            self.assertEqual(1, len(found), f"expected exactly one subject in {text!r}, found {found}")
            subjects[label].add(found[0])
        self.assertEqual(subjects[0], subjects[1])
        self.assertEqual(set(_SUBJECTS), subjects[0])

    def test_asking_for_more_than_exists_says_the_cap(self):
        with self.assertRaises(DatasetError) as caught:
            synthetic(1000)
        self.assertIn("280", str(caught.exception))

class TestGroups(TestCase):
    """Groups exist for one reason: a contrast pair must not straddle a split"""

    def _paired(self, pairs: int = 12) -> LabeledPrompts:
        return LabeledPrompts(
            texts=[f"{side} {index}" for index in range(pairs) for side in ("positive", "negative")],
            labels=[label for _ in range(pairs) for label in (1, 0)],
            name="pairs",
            groups=[index for index in range(pairs) for _ in (0, 1)],
        )

    def test_a_group_id_per_row_is_required(self):
        with self.assertRaises(DatasetError):
            LabeledPrompts(texts=["a", "b"], labels=[1, 0], groups=[0])

    def test_units_bundle_the_rows_a_split_must_keep_together(self):
        self.assertEqual([[0, 1], [2, 3]], self._paired(2).units)

    def test_no_pair_is_ever_split(self):
        """The bug this test exists for: two sentences differing by one word, one
        in train and one in test, and an AUC that is measuring that word"""
        data = self._paired()
        for seed in range(8):
            train, test = data.split(0.25, seed=seed)
            with self.subTest(seed=seed):
                self.assertEqual(set(), set(train.groups) & set(test.groups))
                self.assertEqual(len(data), len(train) + len(test))

    def test_a_split_of_pairs_stays_balanced_on_both_sides(self):
        """Each pair carries one of each label, so whole pairs cannot unbalance it"""
        train, test = self._paired().split(0.25, seed=0)
        self.assertEqual(0.5, train.balance)
        self.assertEqual(0.5, test.balance)

    def test_singletons_and_pairs_are_stratified_apart(self):
        """Otherwise an unlucky shuffle sends every unpaired example one way"""
        data = LabeledPrompts(
            texts=[f"text {index}" for index in range(12)],
            labels=[1, 0] * 6,
            groups=[0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
        )
        train, test = data.split(0.4, seed=0)
        for part in (train, test):
            sizes = [len(unit) for unit in part.units]
            self.assertIn(2, sizes)
            self.assertIn(1, sizes)

    def test_duplicates_are_reported_because_they_leak(self):
        data = LabeledPrompts(texts=["a", "b", "a"], labels=[1, 0, 1])
        self.assertEqual(["a"], data.duplicates)
        self.assertEqual([], synthetic(40).duplicates)

class TestLabelNames(TestCase):
    def test_they_survive_a_subset_and_a_split(self):
        data = LabeledPrompts(texts=list("abcd"), labels=[1, 0, 1, 0], label_names=("honest", "deceptive"))
        train, test = data.split(0.5, seed=0)
        self.assertEqual(("honest", "deceptive"), train.label_names)
        self.assertEqual(("honest", "deceptive"), test.label_names)

    def test_two_names_are_required_and_they_must_differ(self):
        for names in (("only",), ("a", "b", "c"), ("same", "same")):
            with self.subTest(names=names), self.assertRaises(DatasetError):
                LabeledPrompts(texts=["a"], labels=[1], label_names=names)

class TestLoadJsonl(TestCase):
    def _write(self, rows) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write("\n".join(json.dumps(row) for row in rows))
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

    def test_saving_and_loading_are_inverses(self):
        data = LabeledPrompts(texts=["good", "bad"], labels=[1, 0], name="round-trip")
        path = self._write([])
        save_jsonl(data, path)
        again = load_jsonl(path)
        self.assertEqual(data.texts, again.texts)
        self.assertEqual(data.labels, again.labels)

class TestLoadCsv(TestCase):
    """A downloaded contrast set, and the pairing that must survive the import"""

    HEADER = "statement,label,city\n"
    PAIRS = (
        "The city of Lodz is in Poland.,1,Lodz\n"
        "The city of Lodz is in Chad.,0,Lodz\n"
        "The city of Maracay is in Venezuela.,1,Maracay\n"
        "The city of Maracay is in Chad.,0,Maracay\n"
    )

    def _write(self, body: str) -> str:
        path = Path(tempfile.mkdtemp()) / "downloaded.csv"
        path.write_text(body)
        return str(path)

    def test_a_named_group_column_becomes_groups(self):
        data = load_csv(self._write(self.HEADER + self.PAIRS), text_field="statement", group_field="city")
        self.assertEqual([0, 0, 1, 1], data.groups)
        self.assertEqual(2, len(data.units))

    def test_without_a_group_column_the_rows_are_independent(self):
        data = load_csv(self._write(self.HEADER + self.PAIRS), text_field="statement")
        self.assertIsNone(data.groups)

    def test_a_group_split_across_the_file_is_refused(self):
        scattered = (
            "The city of Lodz is in Poland.,1,Lodz\n"
            "The city of Maracay is in Venezuela.,1,Maracay\n"
            "The city of Lodz is in Chad.,0,Lodz\n"
        )
        with self.assertRaises(DatasetError) as caught:
            load_csv(self._write(self.HEADER + scattered), text_field="statement", group_field="city")
        self.assertIn(":4", str(caught.exception))

    def test_a_missing_column_is_named(self):
        with self.assertRaises(DatasetError) as caught:
            load_csv(self._write(self.HEADER + self.PAIRS), text_field="prompt")
        self.assertIn("prompt", str(caught.exception))

    def test_a_label_that_is_not_binary_names_the_line(self):
        body = self.HEADER + "The city of Lodz is in Poland.,yes,Lodz\n"
        with self.assertRaises(DatasetError) as caught:
            load_csv(self._write(body), text_field="statement")
        self.assertIn(":2", str(caught.exception))

    def test_pairs_survive_a_round_trip_through_prompts(self):
        """The reason groups must be adjacent: dumps writes them as an indented run"""
        data = load_csv(self._write(self.HEADER + self.PAIRS), text_field="statement", group_field="city")
        again = parse(dumps(data), name="again")
        self.assertEqual(data.texts, again.texts)
        self.assertEqual(data.labels, again.labels)
        self.assertEqual(data.groups, again.groups)
