import tempfile
from pathlib import Path
from unittest import TestCase

from src.data.dataset import DatasetError, LabeledPrompts
from src.data.prompts import dumps, load_labeled, load_prompts, parse, save_prompts

"""
The format is only worth having if a malformed file says which line is wrong
and a well-formed one means exactly what it looks like it means. So most of
what is tested here is the errors, and the rest is the two properties the
rest of the framework leans on: an indented line joins the group above it,
and whitespace at the end of a prompt is never accidental.
"""

SET = """\
# a comment
name: demo
labels: bad, good

+ I loved it
- I hated it

+ The Golden Gate Bridge is lovely
  - The bridge is lovely
"""

class TestParse(TestCase):
    def setUp(self):
        self.data = parse(SET)

    def test_the_sigil_is_the_label(self):
        self.assertEqual(["I loved it", "I hated it", "The Golden Gate Bridge is lovely", "The bridge is lovely"], self.data.texts)
        self.assertEqual([1, 0, 1, 0], self.data.labels)

    def test_the_header_names_the_set_and_its_labels(self):
        self.assertEqual("demo", self.data.name)
        self.assertEqual(("bad", "good"), self.data.label_names)

    def test_an_indented_line_joins_the_example_above_it(self):
        self.assertEqual([0, 1, 2, 2], self.data.groups)
        self.assertEqual([[0], [1], [2, 3]], self.data.units)

    def test_a_file_without_indentation_has_no_groups(self):
        """Ungrouped is the common case and must stay indistinguishable from before"""
        self.assertIsNone(parse("+ a\n- b\n").groups)

    def test_a_file_with_only_a_header_is_refused(self):
        with self.assertRaises(DatasetError):
            parse("name: empty\n")

class TestHeader(TestCase):
    def test_an_unknown_header_names_the_line(self):
        with self.assertRaises(DatasetError) as caught:
            parse("name: demo\nnmae: typo\n\n+ a\n- b\n")
        self.assertIn(":2", str(caught.exception))
        self.assertIn("nmae", str(caught.exception))

    def test_a_header_after_the_first_example_is_refused(self):
        """Otherwise half the file is described by one name and half by another"""
        with self.assertRaises(DatasetError) as caught:
            parse("+ a\n- b\nname: late\n")
        self.assertIn(":3", str(caught.exception))

    def test_a_header_set_twice_is_refused(self):
        """The second would win silently, and the file would claim two names"""
        with self.assertRaises(DatasetError) as caught:
            parse("name: one\nname: two\n\n+ a\n- b\n")
        self.assertIn(":2", str(caught.exception))

    def test_labels_takes_exactly_two_names(self):
        for value in ("only-one", "a, b, c", "a,"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                parse(f"labels: {value}\n\n+ a\n- b\n")

class TestBadLines(TestCase):
    def test_a_missing_space_after_the_label_names_the_line(self):
        with self.assertRaises(DatasetError) as caught:
            parse("+ fine\n-nospace\n")
        self.assertIn(":2", str(caught.exception))

    def test_a_label_with_no_prompt_is_refused(self):
        with self.assertRaises(DatasetError) as caught:
            parse("+ fine\n+ \n")
        self.assertIn(":2", str(caught.exception))

    def test_a_line_that_is_neither_is_refused(self):
        with self.assertRaises(DatasetError) as caught:
            parse("+ fine\nwhat is this\n")
        self.assertIn(":2", str(caught.exception))

    def test_the_first_example_cannot_be_indented(self):
        """An indented line joins the one above it, and there is nothing above it"""
        with self.assertRaises(DatasetError) as caught:
            parse("  + orphan\n")
        self.assertIn(":1", str(caught.exception))

class TestEscapes(TestCase):
    def test_escapes_become_the_characters_they_name(self):
        data = parse("+ line\\none\n- a\\tb\n+ back\\\\slash\n- x")
        self.assertEqual(["line\none", "a\tb", "back\\slash", "x"], data.texts)

    def test_trailing_whitespace_is_stripped_but_an_escaped_space_survives(self):
        """A prompt ending in a space tokenizes differently, so it has to be visible"""
        data = parse("+ The capital of France is   \n- The capital of France is\\s\n")
        self.assertEqual("The capital of France is", data.texts[0])
        self.assertEqual("The capital of France is ", data.texts[1])

    def test_an_unknown_escape_is_a_typo_and_says_so(self):
        with self.assertRaises(DatasetError) as caught:
            parse("+ ok\n- what\\q\n")
        self.assertIn(":2", str(caught.exception))

    def test_a_lone_trailing_backslash_is_refused(self):
        with self.assertRaises(DatasetError):
            parse("+ ok\n- trailing\\\n")

class TestRoundTrip(TestCase):
    def _round_trip(self, data: LabeledPrompts) -> LabeledPrompts:
        return parse(dumps(data), name=data.name)

    def test_prompts_groups_and_label_names_all_survive(self):
        data = parse(SET)
        again = self._round_trip(data)
        self.assertEqual(data.texts, again.texts)
        self.assertEqual(data.labels, again.labels)
        self.assertEqual(data.groups, again.groups)
        self.assertEqual(data.label_names, again.label_names)
        self.assertEqual(data.name, again.name)

    def test_awkward_whitespace_survives(self):
        """The round trip is only worth anything if it holds for the hard cases"""
        data = LabeledPrompts(
            texts=["trailing ", " leading", "two\nlines", "a\tb", "back\\slash", "  both  "],
            labels=[1, 0, 1, 0, 1, 0],
            name="awkward",
        )
        self.assertEqual(data.texts, self._round_trip(data).texts)

class TestLimit(TestCase):
    def test_limit_rounds_up_to_a_whole_group(self):
        """Half a contrast pair is worse than one pair too many"""
        text = "".join(f"+ positive {index}\n  - negative {index}\n" for index in range(4))
        data = parse(text, limit=3)
        self.assertEqual(4, len(data))
        self.assertEqual([[0, 1], [2, 3]], data.units)

    def test_a_limit_below_one_is_refused(self):
        with self.assertRaises(DatasetError):
            parse(SET, limit=0)

class TestFiles(TestCase):
    def _write(self, text: str, suffix: str = ".prompts") -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        handle.write(text)
        handle.close()
        return handle.name

    def test_the_filename_names_the_set_when_the_header_does_not(self):
        path = self._write("+ a\n- b\n")
        self.assertEqual(Path(path).stem, load_prompts(path).name)

    def test_the_header_wins_over_the_filename(self):
        self.assertEqual("stated", load_prompts(self._write("name: stated\n\n+ a\n- b\n")).name)

    def test_errors_name_the_file_and_the_line(self):
        path = self._write("+ a\nbroken\n")
        with self.assertRaises(DatasetError) as caught:
            load_prompts(path)
        self.assertIn(f"{path}:2", str(caught.exception))

    def test_a_missing_file_is_refused(self):
        with self.assertRaises(DatasetError):
            load_prompts(str(Path(tempfile.gettempdir()) / "definitely-not-here.prompts"))

    def test_save_and_load_are_inverses(self):
        path = self._write("")
        save_prompts(parse(SET), path)
        self.assertEqual(parse(SET).texts, load_prompts(path).texts)

    def test_the_suffix_picks_the_reader(self):
        prompts = self._write("+ a\n- b\n")
        jsonl = self._write('{"text": "a", "label": 1}\n{"text": "b", "label": 0}\n', suffix=".jsonl")
        self.assertEqual(["a", "b"], load_labeled(prompts).texts)
        self.assertEqual(["a", "b"], load_labeled(jsonl).texts)

    def test_an_unknown_suffix_says_which_are_known(self):
        with self.assertRaises(DatasetError) as caught:
            load_labeled(self._write("+ a\n- b\n", suffix=".csv"))
        self.assertIn(".jsonl", str(caught.exception))

class TestShippedSets(TestCase):
    """The files in data/ are documentation, so they have to stay parseable"""

    def test_every_shipped_set_loads_and_is_balanced(self):
        # found relative to this file, so the tests do not depend on the working directory
        for path in sorted((Path(__file__).resolve().parents[1] / "data").glob("*.prompts")):
            with self.subTest(path=str(path)):
                data = load_prompts(str(path))
                self.assertGreater(len(data), 1)
                self.assertEqual([], data.duplicates)
                self.assertAlmostEqual(0.5, data.balance, places=2)
