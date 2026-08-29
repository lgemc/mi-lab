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
from src.ie.views.grid import Grid

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
        """Driven by the actual keypress: enter reaches a view by a route worth testing

        The first version of this called on_enter_row directly and passed while
        pressing enter in a real terminal did nothing at all -- the focused
        DataTable binds enter to its own action and consumes it.
        """
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=_outputs(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()   # the table is focused a frame after the mount
                    view = app.current()
                    self.assertEqual(len(view.visible), 1)
                    await pilot.press("enter")
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

    def test_every_key_the_footer_advertises_actually_fires(self):
        """A binding that is shadowed by a focused widget still shows in the footer"""
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=_outputs(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()   # the table is focused a frame after the mount
                    await pilot.press("y")            # the record behind the row
                    await pilot.pause()
                    opened = app.stack.breadcrumb()
                    await pilot.press("escape")       # back out of it
                    await pilot.pause()
                    await pilot.press("r")            # refresh in place
                    await pilot.pause()
                    return opened, app.stack.breadcrumb(), app.last_flash
        opened, crumb, flash = _run(drive())
        self.assertIn("runs/", opened)
        self.assertEqual(crumb, "runs")
        self.assertIn("reloaded", flash)

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

class TestReachingTheNumbers(TestCase):
    """A run has to reach what it produced, and an artifact has to show its grids

    These are the paths that were missing: a circuit run wrote a .mia beside its
    run.json and the explorer could list the run, list the artifact, and never
    connect the two -- and the tensor view described a layer-by-head grid
    without ever showing one.
    """

    def _with_artifact(self, directory: str) -> str:
        """A run directory holding a run and the circuit artifact it shipped"""
        import torch

        from src.core.config import ModelConfig
        from src.share import storage
        from src.share.schema.artifact import Artifact
        from src.share.schema.metric import Metric
        from src.share.schema.model import ModelRef
        from src.share.schema.node import Node
        from src.share.schema.payload import Payload
        from src.share.schema.site import Site
        from src.share.schema.span import Span
        from src.share.schema.vocabulary import Component, Kind, NodeComponent, Position

        root = _outputs(directory)
        run = next(Path(root).rglob("run.json")).parent
        cfg = ModelConfig(id="tiny", backend="transformers", hf_name="none/tiny",
                          n_layers=4, d_model=6, n_heads=3)
        storage.save(Artifact(
            kind=Kind.CIRCUIT, id="demo-circuit", model=ModelRef.from_config(cfg),
            site=Site.at(range(4), 4, component=Component.HEAD_OUT, position=Position.ALL),
            span=Span("logit_difference", 3.0, 0.5),
            metrics={"faithfulness": Metric(0.9, "recovery under restoration", "recovery")},
            nodes=[Node(id="L2H1", component=NodeComponent.HEAD, layer=2, head=1,
                        in_circuit=True, scores={"attribution": 2.6, "causal": 0.23})],
            tensors={
                "head_attribution": Payload(torch.zeros(4, 3), ["layer", "head"], "logits"),
                "head_effects": Payload(torch.arange(12).float().reshape(4, 3),
                                        ["layer", "head"], "recovery"),
            },
        ), str(run / "circuit.mia"))
        return root

    def test_a_run_reaches_the_artifact_it_shipped(self):
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=self._with_artifact(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    await pilot.press("a")
                    await pilot.pause()
                    return app.stack.breadcrumb(), app.session.artifact.id
        crumb, artifact = _run(drive())
        self.assertEqual(crumb, "runs > nodes")
        self.assertEqual(artifact, "demo-circuit")

    def test_a_run_that_shipped_nothing_says_so_rather_than_nothing(self):
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=_outputs(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    await pilot.press("a")
                    await pilot.pause()
                    return app.last_flash
        self.assertIn("wrote no", _run(drive()))

    def test_a_tensor_drills_into_its_actual_numbers(self):
        """The point of an artifact is the grid, not another description of its shape"""
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=self._with_artifact(directory)), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    await pilot.press("a")
                    await pilot.pause()
                    app.run_command("tensors")
                    await pilot.pause()
                    view = app.current()
                    view._table.move_cursor(row=[r.key for r in view.visible].index("head_effects"))
                    await pilot.press("enter")
                    await pilot.pause()
                    grid = app.current()
                    return grid.title, list(grid.columns), [list(row.cells) for row in grid.visible]
        title, columns, rows = _run(drive())
        self.assertEqual(title, "grid head_effects")
        self.assertEqual(columns[0], "layer \\ head")
        self.assertEqual(columns[1:4], ["h0", "h1", "h2"])
        # the site's layers, not the row index: a grid over a subset is not 0..n
        self.assertEqual([row[0] for row in rows], ["L0", "L1", "L2", "L3"])
        self.assertEqual(rows[1][1:4], ["+3.000", "+4.000", "+5.000"])
        self.assertEqual(rows[1][-1], "h2 +5.000")

    def test_the_artifact_keys_the_footer_advertises_work(self):
        """<n> and <t> were in the hints and implemented nowhere"""
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=self._with_artifact(directory)), start="artifacts")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    await pilot.press("n")
                    await pilot.pause()
                    reached = app.current().title
                    await pilot.press("escape")
                    await pilot.pause()
                    await pilot.press("t")
                    await pilot.pause()
                    return reached, app.current().title
        self.assertEqual(_run(drive()), ("nodes", "tensors"))

    def test_a_grid_of_more_than_two_axes_says_so_instead_of_guessing(self):
        import torch

        from src.share.schema.payload import Payload

        grid = Grid(None, None, "role_weights",
                    Payload(torch.zeros(2, 3, 4), ["layer", "head", "role"], "attention"))
        with self.assertRaises(ValueError) as caught:
            grid.rows()
        self.assertIn("3 axes", str(caught.exception))

class TestDuplicateIds(TestCase):
    """Two runs of one experiment ship two artifacts with the same id

    The id is <dataset>-<model>, so it is not unique and was never meant to be.
    A table keyed on it raises DuplicateKey and takes the whole app down on the
    second row -- which is what happened the first time this was pointed at a
    real outputs directory with two ioi-circuit runs in it.
    """

    def _two_runs(self, directory: str) -> str:
        import torch

        from src.core.config import ModelConfig
        from src.share import storage
        from src.share.schema.artifact import Artifact
        from src.share.schema.model import ModelRef
        from src.share.schema.payload import Payload
        from src.share.schema.site import Site
        from src.share.schema.span import Span
        from src.share.schema.vocabulary import Component, Kind, Position

        cfg = ModelConfig(id="tiny", backend="transformers", hf_name="none/tiny",
                          n_layers=4, d_model=6, n_heads=3)
        for run in ("20260826-aaa", "20260827-bbb"):
            where = Path(directory) / "ioi-circuit" / run
            where.mkdir(parents=True)
            (where / "run.json").write_text(json.dumps({
                "run_id": run, "experiment": "ioi-circuit", "kind": "ioi_circuit",
                "spec_hash": "abc", "status": "completed", "params": {}, "metrics": {},
                "produced": [], "created_at": f"{run}Z", "finished_at": None, "error": None,
            }))
            storage.save(Artifact(
                kind=Kind.CIRCUIT, id="ioi-abc-tiny", model=ModelRef.from_config(cfg),
                site=Site.at(range(4), 4, component=Component.HEAD_OUT, position=Position.ALL),
                span=Span("logit_difference", 3.0, 0.5),
                tensors={
                    "head_attribution": Payload(torch.zeros(4, 3), ["layer", "head"], "logits"),
                    "head_effects": Payload(torch.zeros(4, 3), ["layer", "head"], "recovery"),
                },
            ), str(where / "circuit.mia"))
        return directory

    def test_two_artifacts_with_one_id_both_list(self):
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=self._two_runs(directory)), start="artifacts")
                async with app.run_test(size=(170, 40)) as pilot:
                    await pilot.pause()
                    view = app.current()
                    return view.count, [row.cells[-1] for row in view.visible]
        count, where = _run(drive())
        self.assertEqual(count, "2")
        # the run directory is what tells them apart, and it is on the row
        self.assertEqual(where, ["20260826-aaa", "20260827-bbb"])

    def test_a_view_with_repeated_keys_does_not_take_the_app_down(self):
        """The table key is the position, so a view cannot crash it by having two"""
        async def drive():
            with tempfile.TemporaryDirectory() as directory:
                app = Explorer(session=Refuses(root=directory), start="runs")
                async with app.run_test(size=(140, 40)) as pilot:
                    await pilot.pause()
                    view = app.current()
                    view.rows = lambda: [Row(key="same", cells=("a",)), Row(key="same", cells=("b",))]
                    view.reload()
                    await pilot.pause()
                    return view.count
        self.assertEqual(_run(drive()), "2")

class TestCells(TestCase):
    def test_an_enum_renders_as_its_wire_value(self):
        """(str, Enum) formats as 'Kind.CIRCUIT'; the artifacts table shipped that once"""
        from src.share.schema.vocabulary import Component, Kind

        self.assertEqual(cell_text(Kind.CIRCUIT), "circuit")
        self.assertEqual(cell_text(Component.HEAD_OUT), "head_out")

    def test_a_row_keeps_the_object_it_stands_for(self):
        row = Row(key="L9H9", cells=("L9H9", "name mover"), payload={"layer": 9})
        self.assertEqual(row.payload["layer"], 9)
