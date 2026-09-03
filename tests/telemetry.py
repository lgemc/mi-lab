"""Guards on the journal, whose whole promise is that the file is right *now*.

Every test here is about a run that has not finished: a metric readable before
the process exits, a run killed mid-write, a curve read while it is still being
appended to. A journal that is only correct once the run is over is the thing
this replaced.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.telemetry.journal import (
    Journal,
    TelemetryError,
    journals,
    latest,
    progress,
    read_metrics,
    read_run,
    run_id,
    tail,
    to_columns,
)
from src.telemetry.tracking import (
    USER_AGENT,
    Tracker,
    TrackingConfig,
    TrackingError,
    load_tracking,
)
from src.telemetry.tracking import from_mapping as tracking_from_mapping


class TestJournal(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_a_metric_is_on_disk_before_the_run_ends(self):
        """The promise. A buffered write would pass every other test here."""
        journal = Journal(self.root / "a", params={"steps": 10})
        journal.log(0, loss=1.5)
        self.assertEqual([1.5], [row["loss"] for row in read_metrics(self.root / "a")],
                         "a logged metric was not readable until the journal closed")
        journal.finish()

    def test_the_params_are_written_before_the_first_step(self):
        """A run that dies at step 0 still has to say what it was trying to do"""
        journal = Journal(self.root / "b", params={"config": "qwen3-1.7b", "steps": 5})
        self.addCleanup(journal.finish, "failed")
        run = read_run(self.root / "b")
        self.assertEqual("running", run["status"])
        self.assertEqual("qwen3-1.7b", run["params"]["config"])

    def test_a_second_run_in_the_same_directory_is_refused(self):
        """Second-resolution ids: three launches in one second wrote one interleaved file"""
        first = Journal(self.root / "same", params={"steps": 1})
        self.addCleanup(first.finish)
        with self.assertRaises(TelemetryError):
            Journal(self.root / "same", params={"steps": 1})

    def test_a_failure_is_recorded_rather_than_cleaned_up(self):
        with self.assertRaises(RuntimeError), \
                Journal(self.root / "c", params={"steps": 3}) as journal:
            journal.log(0, loss=2.0)
            raise RuntimeError("the driver said no")
        run = read_run(self.root / "c")
        self.assertEqual("failed", run["status"])
        self.assertIn("the driver said no", run["error"])
        self.assertEqual(1, len(read_metrics(self.root / "c")))

    def test_a_torn_final_line_is_a_write_in_progress(self):
        """Reading between a write and its newline must not raise"""
        journal = Journal(self.root / "d", params={"steps": 3})
        journal.log(0, loss=1.0)
        journal.finish()
        with (self.root / "d" / "metrics.jsonl").open("a") as handle:
            handle.write('{"step": 1, "loss": 0.')
        rows = read_metrics(self.root / "d")
        self.assertEqual(1, len(rows), "a partial trailing line should be skipped, not raise")

    def test_a_torn_line_in_the_middle_is_damage(self):
        """Dropping a row out of the middle silently is how a gap gets explained away"""
        journal = Journal(self.root / "e", params={"steps": 3})
        journal.log(0, loss=1.0)
        journal.finish()
        path = self.root / "e" / "metrics.jsonl"
        path.write_text('{"step": 0, "loss": 0.\n{"step": 1, "loss": 2.0}\n')
        with self.assertRaises(TelemetryError):
            read_metrics(path.parent)

    def test_finish_is_safe_to_call_twice(self):
        """The happy path closes it, and the `finally` closes it again"""
        journal = Journal(self.root / "f", params={"steps": 1})
        journal.log(0, loss=1.0)
        journal.finish("completed")
        journal.finish("completed")
        self.assertEqual("completed", read_run(self.root / "f")["status"])

    def test_progress_projects_from_what_was_logged(self):
        journal = Journal(self.root / "g", params={"steps": 100})
        for step in range(10):
            journal.log(step, loss=1.0)
        state = progress(self.root / "g")
        self.assertEqual(9, state["step"])
        self.assertEqual(100, state["total"])
        self.assertAlmostEqual(0.1, state["fraction"])
        journal.finish()

    def test_progress_on_an_empty_run_does_not_divide_by_zero(self):
        journal = Journal(self.root / "h", params={"steps": 100})
        self.addCleanup(journal.finish, "failed")
        state = progress(self.root / "h")
        self.assertIsNone(state["step"])
        self.assertEqual(0.0, state["seconds_per_step"])

class TestReading(TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_columns_keep_every_row_the_same_length(self):
        """A metric that starts halfway through must not shift against the step axis"""
        rows = [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.5, "density": 0.2}]
        columns = to_columns(rows)
        self.assertEqual([None, 0.2], columns["density"])
        self.assertEqual(len(columns["step"]), len(columns["density"]))

    def test_journals_are_found_at_any_depth_newest_first(self):
        for name in ("20260101-000000-a", "20260202-000000-b"):
            Journal(self.root / "nested" / name, params={"steps": 1}).finish()
        found = journals(self.root)
        self.assertEqual(2, len(found))
        self.assertEqual("20260202-000000-b", found[0].name)
        self.assertEqual(found[0], latest(self.root))

    def test_an_empty_root_has_no_latest(self):
        self.assertIsNone(latest(self.root / "nothing"))
        self.assertEqual([], journals(self.root / "nothing"))

    def test_a_directory_that_is_not_a_journal_says_so(self):
        (self.root / "plain").mkdir()
        with self.assertRaises(TelemetryError):
            read_run(self.root / "plain")
        with self.assertRaises(TelemetryError):
            read_metrics(self.root / "plain")

    def test_tail_returns_the_last_rows(self):
        journal = Journal(self.root / "t", params={"steps": 50})
        for step in range(20):
            journal.log(step, loss=float(step))
        journal.finish()
        self.assertEqual([17, 18, 19], [row["step"] for row in tail(self.root / "t", 3)])

    def test_a_run_id_leads_with_its_timestamp(self):
        """Lexical order is time order, the way find_runs already assumes"""
        first = run_id("alpha")
        self.assertTrue(first[:8].isdigit(), f"{first} does not lead with a date")
        self.assertTrue(first.endswith("-alpha"))

    def test_non_finite_metrics_survive_the_round_trip(self):
        """A diverged loss is exactly when the curve matters most"""
        journal = Journal(self.root / "nan", params={"steps": 2})
        journal.log(0, loss=float("inf"))
        journal.finish()
        raw = (self.root / "nan" / "metrics.jsonl").read_text()
        self.assertIn("Infinity", raw)
        self.assertEqual(float("inf"), json.loads(raw.splitlines()[0])["loss"])

class TestTracking(TestCase):
    """The sink, whose one hard rule is that it may never break the run"""

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_disabled_config_never_touches_the_network(self):
        tracker = Tracker(TrackingConfig(enabled=False, uri="http://unreachable.invalid"))
        self.assertFalse(tracker.active)
        tracker.log(0, {"loss": 1.0})
        tracker.finish()

    def test_an_unreachable_server_disables_rather_than_raises(self):
        """A tracker that kills a two-hour run because a pod restarted is worse
        than no tracker at all"""
        tracker = Tracker(TrackingConfig(enabled=True, uri="http://127.0.0.1:9", timeout=0.2),
                          name="x", params={"a": 1})
        self.assertFalse(tracker.active)
        self.assertIsNotNone(tracker.failure)
        tracker.log(0, {"loss": 1.0})
        tracker.flush()
        tracker.finish()

    def test_a_journal_survives_a_sink_that_explodes(self):
        """The row is on disk before the sink is called, so it stays there"""
        class Exploding:
            def log(self, step, metrics):
                raise RuntimeError("the server is on fire")

        journal = Journal(self.root / "s", params={"steps": 2})
        journal.sink = Exploding()
        with self.assertRaises(RuntimeError):
            journal.log(0, loss=1.0)
        journal.finish("failed")
        self.assertEqual([1.0], [r["loss"] for r in read_metrics(self.root / "s")],
                         "the metric was lost when the sink raised")

    def test_unknown_tracking_keys_are_refused(self):
        with self.assertRaises(TrackingError):
            tracking_from_mapping({"enabled": True, "url": "typo-for-uri"})

    def test_none_disables_without_touching_the_filesystem(self):
        for name in ("none", "off", ""):
            self.assertFalse(load_tracking(name).enabled)

    def test_an_unknown_config_name_names_the_shipped_ones(self):
        with self.assertRaises(TrackingError) as raised:
            load_tracking("nope-not-a-config")
        self.assertIn("mlflow", str(raised.exception))

    def test_the_shipped_mlflow_config_parses_and_carries_no_secret(self):
        config = load_tracking("mlflow")
        self.assertTrue(config.enabled)
        self.assertTrue(config.uri.startswith("https://"))
        raw = (Path(__file__).resolve().parents[1] / "configs" / "tracking"
               / "mlflow.yaml").read_text().lower()
        for leak in ("password", "secret", "token", "api_key"):
            self.assertNotIn(f"{leak}:", raw, f"the committed config carries a {leak}")

    def test_the_request_carries_an_identifying_user_agent(self):
        """Cloudflare 403s `Python-urllib/3.x` as bot traffic, which reads as an
        auth failure and is not one"""
        self.assertNotIn("urllib", USER_AGENT.lower())
        self.assertTrue(USER_AGENT.startswith("mi-lab"))

    def test_non_numeric_metrics_are_skipped_not_sent_as_junk(self):
        tracker = Tracker(TrackingConfig(enabled=False))
        tracker.active = True
        tracker.config = TrackingConfig(enabled=True, uri="http://127.0.0.1:9", flush_every=10_000)
        tracker.log(0, {"loss": 1.0, "note": "hello", "flag": True, "step": 3})
        self.assertEqual({"loss", "step"}, {row["key"] for row in tracker._buffer})
