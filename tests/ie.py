import asyncio
import json
import tempfile
from pathlib import Path
from unittest import TestCase

from src.ie import registry
from src.ie.app import Explorer
from src.ie.command import parse
from src.ie.session import Session
from src.ie.stack import PageStack
from src.ie.view import Row, View, cell_text

"""
The explorer is tested without a checkpoint, because everything structural
about it is answerable without one: what a command means, what the page stack
does when you drill in and go back, and whether a view that needs no model
keeps its promise not to load one.

That last one is the test worth having. `ie` opens on runs and artifacts
instantly only for as long as nothing on that path touches the weights, and
the way that breaks is a view quietly reaching for session.adapter() -- so the
session used here raises if anything does.

The views that do need a model are covered by driving them against a session
that refuses to load, which checks the failure is reported rather than fatal.
"""

class Refuses(Session):
    """A session that fails if anything asks it for weights"""

    def adapter(self):
        raise AssertionError("this view loaded a checkpoint and should not have")

def _run(coroutine):
    return asyncio.run(coroutine)

def _outputs(directory: str) -> str:
    """A root holding one finished run, which is what the listings read"""
    run = Path(directory) / "demo-exp" / "2026-01-01T00-00-00-abc"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({
        "run_id": "2026-01-01T00-00-00-abc", "experiment": "demo-exp", "kind": "ioi_circuit",
        "spec_hash": "deadbeef", "status": "finished", "params": {"size": 8},
        "metrics": {"faithfulness": 0.919}, "produced": [],
        "created_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:04:00Z", "error": None,
    }))
    return directory

class TestCommands(TestCase):
    def test_an_alias_is_the_resource(self):
        self.assertEqual(parse(":ar").resource, "artifacts")
        self.assertEqual(parse("runs").resource, "runs")

    def test_a_filter_rides_along_in_either_spelling(self):
        self.assertEqual(parse(":runs failed").filter, "failed")
        self.assertEqual(parse(":runs /failed").filter, "failed")
        self.assertEqual(parse("ru /failed").resource, "runs")

    def test_a_setter_keeps_the_whole_line(self):
        """A prompt has spaces and slashes in it, so splitting on either loses it"""
        command = parse(":prompt Then, Jack and Mary went to the store/shop")
        self.assertTrue(command.is_setter)
        self.assertEqual(command.argument, "Then, Jack and Mary went to the store/shop")

    def test_an_empty_line_is_not_an_error(self):
        """Pressing colon and then escape is a thing people do constantly"""
        self.assertIsNone(parse(":"))
        self.assertIsNone(parse("   "))

    def test_every_registered_view_is_reachable_by_name(self):
        for name in registry.VIEWS:
            self.assertIsNotNone(registry.build(name, None, Session()))
        self.assertIsNone(registry.build("nope", None, Session()))

class Page(View):
    """A view with fixed rows, for testing the stack rather than a data source"""

    def __init__(self, title: str, rows=()):
        super().__init__(app_ref=None, session=None)
        self.title = title
        self._fixed = list(rows)

    def rows(self):
        return self._fixed

    def on_show(self):
        pass

class TestPageStack(TestCase):
    def test_drilling_in_and_going_back(self):
        stack = PageStack()
        stack.push(Page("runs"))
        stack.push(Page("run detail"))
        self.assertEqual(stack.breadcrumb(), "runs > run detail")
        stack.pop()
        self.assertEqual(stack.breadcrumb(), "runs")

    def test_the_same_page_replaces_rather_than_stacks(self):
        stack = PageStack()
        stack.push(Page("runs"))
        stack.push(Page("runs"))
        self.assertEqual(len(stack), 1)

    def test_two_pages_of_one_kind_are_different_pages(self):
        """activations and activations L9 are the layer list and one layer's positions"""
        stack = PageStack()
        stack.push(Page("activations"))
        stack.push(Page("activations L9"))
        self.assertEqual(len(stack), 2)

    def test_the_root_cannot_be_popped(self):
        """Popping the last view would leave a screen with no way to say what to do"""
        stack = PageStack()
        stack.push(Page("runs"))
        self.assertIsNone(stack.pop())
        self.assertEqual(len(stack), 1)

class TestExplorer(TestCase):
    def test_runs_and_artifacts_never_load_a_checkpoint(self):
        """The laziness is the whole reason ie opens instantly; a view can break it"""
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=_outputs(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    for resource in ("runs", "artifacts", "models", "help"):
                        app.run_command(resource)
                        await pilot.pause()
                    return app.current().title
        self.assertEqual(_run(drive()), "help")

    def test_a_run_shows_up_and_drills_into_its_record(self):
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=_outputs(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    view = app.current()
                    self.assertEqual(len(view.visible), 1)
                    view.on_enter_row(view.visible[0])
                    await pilot.pause()
                    return app.stack.breadcrumb(), dict(app.current().record)
        crumb, record = _run(drive())
        self.assertIn("run 2026-01-01T00-00-00-abc", crumb)
        self.assertEqual(record["status"], "finished")
        self.assertEqual(record["metric.faithfulness"], 0.919)

    def test_a_view_that_cannot_load_reports_it_instead_of_dying(self):
        """Offline is the normal case for a laptop, and it must not take the app down"""
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                session = Refuses(root=directory, model_id="gpt2-small")
                app = Explorer(session=session, start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    app.run_command("tokens")
                    await pilot.pause()
                    return app.current().title, app.last_flash
        title, flash = _run(drive())
        self.assertEqual(title, "runs")
        self.assertIn("cannot load", flash)

    def test_an_unknown_resource_names_the_known_ones(self):
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=directory), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    app.run_command("pods")
                    await pilot.pause()
                    return app.last_flash
        self.assertIn("artifacts", _run(drive()))

    def test_a_filter_narrows_and_escape_clears_it(self):
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=_outputs(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    view = app.current()
                    view.set_filter("nothing-matches-this")
                    await pilot.pause()
                    narrowed = view.count
                    app.action_back()
                    await pilot.pause()
                    return narrowed, view.count, app.stack.breadcrumb()
        narrowed, restored, crumb = _run(drive())
        self.assertEqual(narrowed, "0/1")
        self.assertEqual(restored, "1")
        self.assertEqual(crumb, "runs")

class TestCells(TestCase):
    def test_an_enum_renders_as_its_wire_value(self):
        """(str, Enum) formats as 'Kind.CIRCUIT'; the artifacts table shipped that once"""
        from src.share.schema.vocabulary import Component, Kind

        self.assertEqual(cell_text(Kind.CIRCUIT), "circuit")
        self.assertEqual(cell_text(Component.HEAD_OUT), "head_out")

    def test_a_row_keeps_the_object_it_stands_for(self):
        row = Row(key="L9H9", cells=("L9H9", "name mover"), payload={"layer": 9})
        self.assertEqual(row.payload["layer"], 9)
