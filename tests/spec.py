import tempfile
from pathlib import Path
from unittest import TestCase

from src.core.dataset import DatasetError
from src.core.spec import ExperimentSpec, ModelSpec, SpecError, load_spec, save_spec

"""
The spec is the reproducibility story, so what is tested here is mostly about
the hash: that it covers everything which changes a number, that it ignores
everything which does not, and that a key nobody knows about is an error
rather than a line that quietly did nothing.
"""

def _write(text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name

class TestComposition(TestCase):
    def test_defaults_load_without_a_file(self):
        spec = load_spec()
        self.assertEqual("probe_sweep", spec.kind)
        self.assertEqual("gpt2-small", spec.model.config)

    def test_a_file_overrides_only_what_it_mentions(self):
        path = _write("experiment: mine\nmodel:\n  config: pythia-70m\n")
        spec = load_spec(path)
        self.assertEqual("mine", spec.experiment)
        self.assertEqual("pythia-70m", spec.model.config)
        self.assertEqual(load_spec().method.epochs, spec.method.epochs)

    def test_dotted_overrides_beat_the_file(self):
        path = _write("model:\n  config: pythia-70m\n")
        spec = load_spec(path, overrides=["model.config=gpt2-medium", "method.lr=0.5"])
        self.assertEqual("gpt2-medium", spec.model.config)
        self.assertEqual(0.5, spec.method.lr)

    def test_a_key_outside_the_schema_is_an_error(self):
        """A typo in a spec must not be a line that silently did nothing"""
        with self.assertRaises(SpecError):
            load_spec(overrides=["methodd.lr=0.5"])
        with self.assertRaises(SpecError):
            load_spec(_write("method:\n  learning_rate: 0.5\n"))

    def test_a_wrongly_typed_value_is_an_error(self):
        with self.assertRaises(SpecError):
            load_spec(overrides=["method.epochs=many"])

    def test_the_file_name_appears_in_the_error(self):
        path = _write("nonsense: 1\n")
        with self.assertRaises(SpecError) as caught:
            load_spec(path)
        self.assertIn(path, str(caught.exception))

    def test_a_saved_spec_reloads_identically(self):
        spec = load_spec(overrides=["experiment=round-trip", "method.fracs=[0.1,0.9]"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "spec.yaml")
            save_spec(spec, path)
            self.assertEqual(spec.spec_hash, load_spec(path).spec_hash)

class TestSpecHash(TestCase):
    def test_it_is_stable_across_calls(self):
        self.assertEqual(load_spec().spec_hash, load_spec().spec_hash)

    def test_output_paths_do_not_change_it(self):
        """Writing the same experiment elsewhere is the same experiment"""
        baseline = load_spec().spec_hash
        self.assertEqual(baseline, load_spec(overrides=["output.root=/tmp/elsewhere"]).spec_hash)
        self.assertEqual(baseline, load_spec(overrides=["output.save_probe=false"]).spec_hash)

    def test_anything_that_changes_a_number_changes_it(self):
        baseline = load_spec().spec_hash
        for override in (
            "model.config=pythia-70m", "data.size=50", "data.test_frac=0.4",
            "method.kind=difference_of_means", "method.lr=0.9", "method.fracs=[0.5]", "seed=1",
        ):
            with self.subTest(override=override):
                self.assertNotEqual(baseline, load_spec(overrides=[override]).spec_hash)

class TestModelSpec(TestCase):
    def test_unset_overrides_leave_the_config_alone(self):
        from src.core.config import load_config

        self.assertEqual(load_config("gpt2-small"), ModelSpec(config="gpt2-small").resolve())

    def test_stated_overrides_are_applied(self):
        resolved = ModelSpec(config="gpt2-small", batch_size=2, device="cpu").resolve()
        self.assertEqual(2, resolved.batch_size)
        self.assertEqual("gpt2-small", resolved.id)

class TestDataSpec(TestCase):
    def test_synthetic_is_the_default_and_needs_no_file(self):
        data = load_spec(overrides=["data.size=20"]).data.load(seed=0)
        self.assertEqual(20, len(data))

    def test_jsonl_without_a_path_is_refused(self):
        with self.assertRaises(SpecError):
            load_spec(overrides=["data.source=jsonl"])

    def test_an_unknown_source_says_what_is_known(self):
        spec = load_spec()
        spec.data.source = "parquet"
        with self.assertRaises(SpecError) as caught:
            spec.data.load()
        self.assertIn("synthetic", str(caught.exception))

    def test_a_missing_jsonl_is_refused_by_the_loader(self):
        spec = load_spec(overrides=["data.source=jsonl", "data.path=/tmp/not-here.jsonl"])
        with self.assertRaises(DatasetError):
            spec.data.load()

class TestValidation(TestCase):
    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(SpecError):
            load_spec(overrides=["method.kind=svm"])

    def test_fracs_must_be_fractions(self):
        for fracs in ("[8]", "[-0.1]", "[]"):
            with self.subTest(fracs=fracs), self.assertRaises(SpecError):
                load_spec(overrides=[f"method.fracs={fracs}"])

    def test_a_nonsense_test_fraction_is_refused(self):
        with self.assertRaises(SpecError):
            load_spec(overrides=["data.test_frac=1.5"])

class TestShippedSpecs(TestCase):
    def test_every_spec_in_specs_loads(self):
        paths = sorted(Path("specs").glob("*.yaml"))
        self.assertTrue(paths, "no specs found")
        for path in paths:
            with self.subTest(path=str(path)):
                spec = load_spec(str(path))
                self.assertIsInstance(spec, ExperimentSpec)
                self.assertNotEqual("unnamed", spec.experiment)
