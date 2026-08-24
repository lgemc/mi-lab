import json
import tempfile
from pathlib import Path
from unittest import TestCase

from src.core.run import Ref, Run, RunError, find_runs

"""
Run is plain data, and these tests hold it to that: it must survive a round
trip through JSON unchanged, and a run that failed must still be readable
afterwards, because those are the ones worth looking at.
"""

def _run(**overrides) -> Run:
    fields = dict(experiment="test", kind="probe_train", spec_hash="abc123abc123")
    fields.update(overrides)
    return Run.start(**fields)

class TestLifecycle(TestCase):
    def test_a_new_run_is_running_and_carries_its_hash(self):
        run = _run()
        self.assertEqual("running", run.status)
        self.assertTrue(run.run_id.endswith("abc123abc123"), run.run_id)
        self.assertIsNone(run.finished_at)
        self.assertIsNone(run.duration_seconds)

    def test_metrics_are_coerced_to_floats(self):
        run = _run().record(auc=1, layer=8)
        self.assertEqual({"auc": 1.0, "layer": 8.0}, run.metrics)

    def test_finishing_completes_it(self):
        run = _run().finish()
        self.assertEqual("completed", run.status)
        self.assertIsNone(run.error)
        self.assertGreaterEqual(run.duration_seconds, 0.0)

    def test_a_failure_records_the_reason(self):
        run = _run().finish(error=ValueError("no layers"))
        self.assertEqual("failed", run.status)
        self.assertIn("ValueError", run.error)
        self.assertIn("no layers", run.error)

    def test_produced_artifacts_are_named_by_kind(self):
        run = _run().produce("probe", "probe-layer8.pt")
        self.assertEqual([Ref(kind="probe", id="probe-layer8.pt")], run.produced)

class TestPersistence(TestCase):
    def test_a_saved_run_reloads_identically(self):
        run = _run().record(auc=0.91).produce("probe", "p.pt").finish()
        with tempfile.TemporaryDirectory() as directory:
            run.save(directory)
            self.assertEqual(run, Run.load(directory))

    def test_a_failed_run_is_still_readable(self):
        run = _run().finish(error=RuntimeError("cuda is on fire"))
        with tempfile.TemporaryDirectory() as directory:
            run.save(directory)
            self.assertEqual("failed", Run.load(directory).status)

    def test_saving_creates_the_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = str(Path(directory) / "nested" / "deeper")
            _run().save(target)
            self.assertTrue((Path(target) / "run.json").exists())

    def test_a_missing_run_is_refused(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(RunError):
            Run.load(directory)

    def test_malformed_json_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text("{not json")
            with self.assertRaises(RunError):
                Run.load(directory)

    def test_a_run_missing_its_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "run.json").write_text(json.dumps({"run_id": "x"}))
            with self.assertRaises(RunError):
                Run.load(directory)

class TestFindRuns(TestCase):
    def test_runs_come_back_newest_first(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("20260101-000000-aaa", "20260301-000000-ccc", "20260201-000000-bbb"):
                run = _run()
                run.run_id = name
                run.save(str(Path(root) / name))
            self.assertEqual(
                ["20260301-000000-ccc", "20260201-000000-bbb", "20260101-000000-aaa"],
                [run.run_id for run in find_runs(root)],
            )

    def test_a_half_written_run_does_not_break_the_listing(self):
        """A process killed mid-write should cost you that run, not the listing"""
        with tempfile.TemporaryDirectory() as root:
            _run().save(str(Path(root) / "good"))
            (Path(root) / "broken").mkdir()
            (Path(root) / "broken" / "run.json").write_text("{")
            self.assertEqual(1, len(find_runs(root)))

    def test_a_missing_root_is_empty_rather_than_fatal(self):
        self.assertEqual([], find_runs("/tmp/definitely-no-runs-here"))
