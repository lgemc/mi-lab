"""Guards on the run log, whose job is to say what a run was doing when it died.

Everything here is text a human reads against a budget: a duration that
rounds before it carries (1799.6s is 30m00s, never 29m60s), a budget whose
stop line carries the marker the pipeline re-invokes on, a progress counter
whose last tick always prints so a finished loop never looks truncated, and a
log that lands on disk when a file is set and nowhere when it is unset.
"""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.telemetry.observe import (
    BUDGET_MARKER,
    Budget,
    Progress,
    duration,
    log,
    log_file,
    set_log_file,
    step,
)


class TestDuration(TestCase):
    def test_short_spans_are_seconds(self):
        self.assertEqual("0s", duration(0))
        self.assertEqual("89s", duration(89.4))

    def test_minutes_carry_from_ninety_seconds(self):
        self.assertEqual("1m30s", duration(90))
        self.assertEqual("29m59s", duration(1799))

    def test_rounding_happens_before_the_carry(self):
        """1799.6s is thirty minutes, not 29m60s"""
        self.assertEqual("30m00s", duration(1799.6))

    def test_hours_carry_from_ninety_minutes(self):
        self.assertEqual("1h30m", duration(90 * 60))
        self.assertEqual("2h05m", duration(2 * 3600 + 5 * 60 + 59))

    def test_negative_spans_are_zero(self):
        """A clock that went backwards is not a negative duration"""
        self.assertEqual("0s", duration(-4))


class TestBudget(TestCase):
    def test_an_item_fits_while_the_allowance_covers_it(self):
        budget = Budget(3600)
        self.assertTrue(budget.fits(60))
        self.assertFalse(budget.fits(4000))

    def test_a_spent_budget_fits_nothing(self):
        budget = Budget(0)
        self.assertFalse(budget.fits(0.0))
        self.assertLessEqual(budget.left, 0.0)

    def test_the_stop_line_carries_the_marker_the_pipeline_looks_for(self):
        """`experiment.pipeline` re-invokes a repeat step on exactly this string"""
        line = Budget(600).stop_line(120, item="sweep step")
        self.assertTrue(line.startswith(BUDGET_MARKER))
        self.assertIn("sweep step needs ~2m00s", line)
        self.assertIn("10m00s", line)


class TestLog(TestCase):
    def setUp(self):
        self.addCleanup(set_log_file, None)

    def test_a_log_file_receives_every_line_and_stdout_keeps_it(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            set_log_file(path)
            self.assertEqual(path, log_file())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                log("hello", indent=1)
            # the file opens with a header naming the invocation, then mirrors stdout line for line
            lines = path.read_text().strip().splitlines()
            self.assertTrue(lines[0].startswith("=== "), lines)
            self.assertTrue(lines[-1].endswith("]   hello"), lines)
            self.assertEqual(lines[-1] + "\n", out.getvalue())

    def test_unsetting_the_file_stops_writing_to_it(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            set_log_file(path)
            with contextlib.redirect_stdout(io.StringIO()):
                log("first")
                set_log_file(None)
                log("second")
            self.assertIsNone(log_file())
            self.assertIn("first", path.read_text())
            self.assertNotIn("second", path.read_text())

    def test_a_step_that_raises_still_says_which_step_it_was(self):
        """The failure mode this exists for: a traceback twenty minutes in with no pass named"""
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(RuntimeError), step("capture means"):
            raise RuntimeError("driver gone")
        text = out.getvalue()
        self.assertIn("-> capture means", text)
        self.assertIn("!! capture means FAILED", text)
        self.assertIn("RuntimeError: driver gone", text)

    def test_a_step_reports_the_facts_it_was_handed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), step("score") as facts:
            facts["bleu"] = 31.2
        self.assertIn("<- score in", out.getvalue())
        self.assertIn("bleu 31.2", out.getvalue())


class TestProgress(TestCase):
    def ticks(self, total: int, every: int) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            progress = Progress(total, "items", every=every)
            for _ in range(total):
                progress.tick()
        return out.getvalue()

    def test_every_throttles_the_printing(self):
        lines = self.ticks(10, every=5).strip().splitlines()
        self.assertEqual(2, len(lines))
        self.assertIn("items 5/10", lines[0])
        self.assertIn("items 10/10", lines[1])

    def test_the_last_tick_always_prints(self):
        """A loop of 7 at every=5 would otherwise end on 5/7 and look truncated"""
        lines = self.ticks(7, every=5).strip().splitlines()
        self.assertEqual(2, len(lines))
        self.assertIn("items 7/7", lines[-1])

    def test_finish_returns_the_elapsed_seconds(self):
        with contextlib.redirect_stdout(io.StringIO()):
            elapsed = Progress(3, "items").finish()
        self.assertGreaterEqual(elapsed, 0.0)
