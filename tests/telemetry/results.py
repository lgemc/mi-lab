"""Guards on the results root: one model per directory, and progress that survives a restart.

The root is read from the environment on every call rather than cached at
import, which is what lets a pipeline set it per step and lets these tests
point it at a temporary directory. The guard is the rule that keeps a
directory named for its phase from mixing two models' numbers, and the
refusal has to name the way out.
"""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.telemetry.results import (
    ENV_ROOT,
    STAMP,
    ResultsError,
    guard,
    load_state,
    merge_section,
    owner,
    result,
    root,
    save_state,
)


class TemporaryRoot(TestCase):
    """A fresh results root per test, restored to whatever the shell had afterwards"""

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "results"
        previous = os.environ.get(ENV_ROOT)
        os.environ[ENV_ROOT] = str(self.root)

        def restore():
            if previous is None:
                os.environ.pop(ENV_ROOT, None)
            else:
                os.environ[ENV_ROOT] = previous

        self.addCleanup(restore)


class TestRoot(TemporaryRoot):
    def test_the_root_is_read_from_the_environment_on_every_call(self):
        self.assertEqual(self.root, root())
        os.environ[ENV_ROOT] = str(self.root / "elsewhere")
        self.assertEqual(self.root / "elsewhere", root())

    def test_a_result_lives_under_the_root(self):
        self.assertEqual(self.root / "phase1b-flops-model.json", result("phase1b-flops-model.json"))


class TestGuard(TemporaryRoot):
    def test_the_first_writer_stamps_the_directory(self):
        self.assertIsNone(owner(self.root))
        self.assertEqual(self.root, guard("qwen3-1.7b"))
        self.assertEqual("qwen3-1.7b", owner(self.root))
        self.assertEqual({"config": "qwen3-1.7b"}, json.loads((self.root / STAMP).read_text()))

    def test_the_same_config_passes_again(self):
        guard("qwen3-1.7b")
        guard("qwen3-1.7b")

    def test_another_config_is_refused_with_the_way_out(self):
        """The message has to name the environment variable, not only the problem"""
        guard("qwen3-1.7b")
        with self.assertRaises(ResultsError) as caught:
            guard("qwen3-8b")
        self.assertIn(f"{ENV_ROOT}=results/qwen3-8b", str(caught.exception))
        self.assertIn("qwen3-1.7b", str(caught.exception))


class TestState(TemporaryRoot):
    def test_a_missing_file_yields_a_fresh_copy_of_the_default(self):
        default = {"done": {}}
        state = load_state(self.root / "progress.json", default)
        state["done"]["x"] = 1
        self.assertEqual({"done": {}}, default)

    def test_state_round_trips_and_creates_its_directory(self):
        path = self.root / "deep" / "progress.json"
        save_state(path, {"hypotheses": ["hola", "adiós"]})
        self.assertEqual({"hypotheses": ["hola", "adiós"]}, load_state(path))

    def test_merging_a_section_keeps_the_others(self):
        """Each phase-0 script rewrites only its own key, in any order, any number of times"""
        path = self.root / "report.json"
        merge_section(path, "model_smoke", {"load_seconds": 3})
        merge_section(path, "prompt_form", {"chosen": "few_shot"})
        merged = merge_section(path, "model_smoke", {"load_seconds": 4})
        self.assertEqual({"model_smoke": {"load_seconds": 4}, "prompt_form": {"chosen": "few_shot"}}, merged)
        self.assertEqual(merged, load_state(path))
