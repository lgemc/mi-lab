import tempfile
from pathlib import Path
from unittest import TestCase

from src.core.dataset import DatasetError
from src.core.spec import (
    ExperimentSpec,
    ModelSpec,
    SpecError,
    compose_spec,
    groups,
    load_spec,
    save_spec,
)

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

class TestHydraComposition(TestCase):
    def test_the_groups_beside_the_config_are_discovered(self):
        found = groups()
        self.assertEqual({"data", "method", "model", "preset"}, set(found))
        self.assertIn("gpt2-small", found["model"])

    def test_composing_with_no_overrides_gives_the_defaults(self):
        spec = compose_spec()
        self.assertEqual("gpt2-small", spec.model.config)
        self.assertEqual("logistic", spec.method.kind)

    def test_a_group_can_be_swapped_whole(self):
        spec = compose_spec(overrides=["model=pythia-70m", "method=difference_of_means"])
        self.assertEqual("pythia-70m", spec.model.config)
        self.assertEqual("difference_of_means", spec.method.kind)

    def test_a_single_key_inside_a_group_can_be_overridden(self):
        self.assertEqual(0.2, compose_spec(overrides=["method.lr=0.2"]).method.lr)

    def test_every_preset_composes_and_names_itself(self):
        for preset in groups()["preset"]:
            with self.subTest(preset=preset):
                spec = compose_spec(preset=preset)
                self.assertIsInstance(spec, ExperimentSpec)
                self.assertNotEqual("unnamed", spec.experiment)

    def test_every_model_option_names_a_config_that_exists(self):
        """A group option pointing at a missing config fails only at load time otherwise"""
        from src.core.config import load_config

        for option in groups()["model"]:
            with self.subTest(option=option):
                load_config(compose_spec(overrides=[f"model={option}"]).model.config)

    def test_composing_twice_in_one_process_works(self):
        """Hydra keeps its state in a global singleton, so this is not free"""
        first = compose_spec(overrides=["model=pythia-70m"])
        second = compose_spec(overrides=["model=gpt2-small"])
        self.assertEqual("pythia-70m", first.model.config)
        self.assertEqual("gpt2-small", second.model.config)

    def test_an_unknown_preset_lists_the_ones_that_exist(self):
        with self.assertRaises(SpecError) as caught:
            compose_spec(preset="does-not-exist")
        self.assertIn("sentiment-sweep", str(caught.exception))

    def test_an_unknown_group_option_is_refused(self):
        with self.assertRaises(SpecError):
            compose_spec(overrides=["model=llama-400b"])

    def test_a_key_outside_the_schema_is_refused(self):
        """Hydra's + prefix appends past struct mode, so this needs its own guard

        Without it, `+nonsense=1` composes cleanly, is dropped on the way to
        the dataclass, and never reaches the spec hash -- an override that
        looks like it did something and did not.
        """
        for override in ("+nonsense=1", "+method.learning_rate=0.5", "+model.name=gpt2"):
            with self.subTest(override=override), self.assertRaises(SpecError):
                compose_spec(overrides=[override])

    def test_a_wrongly_typed_override_is_refused(self):
        with self.assertRaises(SpecError):
            compose_spec(overrides=["method.epochs=lots"])

    def test_a_composed_spec_survives_the_round_trip_a_run_depends_on(self):
        """What `run replay` rests on: the saved spec is self-contained

        A run writes a resolved spec.yaml with no groups and no defaults list,
        and must stay reproducible from it after specs/ has moved on.
        """
        spec = compose_spec(preset="sentiment-sweep", overrides=["model=pythia-70m", "seed=3"])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "spec.yaml")
            save_spec(spec, path)
            self.assertEqual(spec.spec_hash, load_spec(path).spec_hash)
