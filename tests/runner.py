import tempfile
from pathlib import Path
from typing import Optional
from unittest import SkipTest, TestCase

from src.experiment.run import Run, find_runs
from src.experiment.runner import EXPERIMENTS, register_experiment, run_directory, run_experiment
from src.experiment.spec import SpecError, load_spec

"""
The runner is tested end to end against a real checkpoint, because the thing
worth checking is exactly the part a fake would skip: that a run directory is
left behind containing the spec that produced it, the numbers it measured, and
the artifact it wrote.

The failure path gets the same treatment. A run that raises must still be on
disk afterwards, marked failed, with the reason recorded.
"""

def _tiny(root: str, overrides: Optional[dict] = None):
    """A spec small enough to run in a unit test, on the cached laptop model

    Keyed by dotted path rather than held as a set of strings: with a set, a
    caller overriding `kind` adds a second `kind=` entry instead of replacing
    the first, and which one wins comes down to iteration order.
    """
    settings = {
        "experiment": "unit-test", "kind": "probe_train", "model.config": "gpt2-small",
        "model.batch_size": 8, "data.size": 20, "method.fracs": "[0.65]",
        "method.epochs": 40, "output.root": root,
    }
    settings.update(overrides or {})
    return load_spec(overrides=[f"{key}={value}" for key, value in settings.items()])

class RunnerTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        # the shared adapter rather than a fresh one: the runner builds its own
        # adapters anyway, and this only asks whether the checkpoint is reachable
        from .stubs.model import shared_adapter

        if shared_adapter() is None:
            raise SkipTest("gpt2-small is not available; run once with network access")

class TestProbeTrain(RunnerTestCase):
    def test_a_completed_run_leaves_everything_needed_to_read_it(self):
        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root)
            run = run_experiment(spec)
            directory = run_directory(spec, run)

            self.assertEqual("completed", run.status)
            self.assertEqual(spec.spec_hash, run.spec_hash)
            self.assertTrue((directory / "run.json").exists())
            self.assertTrue((directory / "spec.yaml").exists())
            self.assertIn("auc", run.metrics)
            self.assertIn("baseline_auc", run.metrics)
            self.assertEqual(run, Run.load(str(directory)))

    def test_the_probe_it_wrote_is_the_probe_it_claims(self):
        from src.methods.probing import LinearProbe

        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root)
            run = run_experiment(spec)
            (ref,) = run.produced
            probe = LinearProbe.load(str(run_directory(spec, run) / ref.id))
            self.assertEqual("gpt2-small", probe.model_id)
            self.assertEqual(int(run.metrics["layer"]), probe.layer)

    def test_the_saved_spec_reproduces_the_hash(self):
        """The point of writing spec.yaml: the run says exactly what it ran"""
        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root)
            run = run_experiment(spec)
            reloaded = load_spec(str(run_directory(spec, run) / "spec.yaml"))
            self.assertEqual(run.spec_hash, reloaded.spec_hash)

    def test_save_probe_false_produces_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            run = run_experiment(_tiny(root, {"output.save_probe": "false"}))
            self.assertEqual([], run.produced)

class TestProbeSweep(RunnerTestCase):
    def test_it_records_a_number_for_every_layer_it_tried(self):
        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root, {"kind": "probe_sweep", "method.fracs": "[0.1,0.5,0.9]"})
            run = run_experiment(spec)
            per_layer = [key for key in run.metrics if key.startswith("auc_layer_")]
            self.assertEqual(3, len(per_layer))
            self.assertEqual(max(run.metrics[key] for key in per_layer), run.metrics["best_auc"])

    def test_difference_of_means_needs_no_hyperparameters(self):
        """It takes no lr or epochs, so passing it any is a TypeError rather than a no-op"""
        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root, {"kind": "probe_sweep", "method.kind": "difference_of_means"})
            self.assertEqual("completed", run_experiment(spec).status)

class TestFailurePath(RunnerTestCase):
    def test_a_failed_run_is_still_on_disk_with_its_reason(self):
        @register_experiment("unit-test-explodes")
        def _explode(spec, run, directory):
            raise RuntimeError("the hook fell off")

        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root, {"kind": "unit-test-explodes"})
            with self.assertRaises(RuntimeError):
                run_experiment(spec)

            # located by its run.json rather than by walking a known depth: the run
            # raised, so there is no Run object to ask run_directory about
            (marker,) = Path(root).rglob("run.json")
            directory = marker.parent
            failed = Run.load(str(directory))
            self.assertEqual("failed", failed.status)
            self.assertIn("the hook fell off", failed.error)
            self.assertTrue((directory / "spec.yaml").exists())
            # and a failed run is a run: it still turns up in the listing
            self.assertEqual(["failed"], [found.status for found in find_runs(root)])
        EXPERIMENTS.pop("unit-test-explodes")

    def test_an_unknown_kind_says_what_is_known(self):
        with tempfile.TemporaryDirectory() as root:
            spec = _tiny(root)
            spec.kind = "telepathy"
            with self.assertRaises(SpecError) as caught:
                run_experiment(spec)
            self.assertIn("probe_train", str(caught.exception))
