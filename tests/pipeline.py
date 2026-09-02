"""Guards on the pipeline runner: the order is a file, and a finished step stays finished.

A pipeline is composed from `pipelines/` with Hydra's grammar and run as
subprocesses, so what is checked here is the part that does not need a model:
the mapping becomes steps in order, `only` refuses a name it does not have, a
dry run executes nothing, a completed step is recorded and skipped on
re-entry, and a step that says it stopped on its budget is invoked again.
"""

import contextlib
import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.experiment.pipeline import (
    PIPELINE_DIR,
    STATE_FILE,
    Pipeline,
    PipelineError,
    Step,
    compose,
    run,
)
from src.telemetry.observe import BUDGET_MARKER, set_log_file
from src.telemetry.results import ENV_ROOT

MAPPING = {
    "run": {
        "name": "demo",
        "config": "gpt2-small",
        "env": {"MI_LAB_EVAL_SENTENCES": 20},
        "steps": [
            {"name": "flops", "module": "scripts.phase1b_flops", "args": [160]},
            {"name": "sweep", "module": "scripts.phase1b_ablation", "args": ["layers"], "repeat": True},
        ],
    },
    "resume": True,
    "dry_run": False,
    "only": [],
}


class TestMapping(TestCase):
    def test_the_mapping_becomes_steps_in_order(self):
        pipeline = Pipeline.from_mapping(MAPPING)
        self.assertEqual(["flops", "sweep"], [step.name for step in pipeline.steps])
        self.assertEqual(["160"], pipeline.steps[0].args)
        self.assertTrue(pipeline.steps[1].repeat)
        self.assertEqual({"MI_LAB_EVAL_SENTENCES": "20"}, pipeline.env)

    def test_a_step_command_names_the_module_and_the_config_first(self):
        step = Step(name="flops", module="scripts.phase1b_flops", args=["160"])
        self.assertEqual([sys.executable, "-m", "scripts.phase1b_flops", "gpt2-small", "160"],
                         step.command("gpt2-small"))
        self.assertEqual("scripts.phase1b_flops:flops", step.marker)

    def test_only_selects_by_name_and_refuses_a_stranger(self):
        pipeline = Pipeline.from_mapping({**MAPPING, "only": ["sweep"]})
        self.assertEqual(["sweep"], [step.name for step in pipeline.selected])
        with self.assertRaises(PipelineError):
            _ = Pipeline.from_mapping({**MAPPING, "only": ["ranking"]}).selected

    def test_a_missing_key_is_named(self):
        with self.assertRaises(PipelineError) as caught:
            Pipeline.from_mapping({"run": {"name": "demo"}})
        self.assertIn("config", str(caught.exception))

    def test_the_root_and_state_follow_the_environment_the_pipeline_sets(self):
        pipeline = Pipeline.from_mapping({**MAPPING, "run": {**MAPPING["run"], "env": {ENV_ROOT: "results/demo"}}})
        self.assertEqual(Path("results/demo"), pipeline.root)
        self.assertEqual(Path("results/demo") / STATE_FILE, pipeline.state_path)


class TestCompose(TestCase):
    def test_every_shipped_run_composes(self):
        for path in sorted(PIPELINE_DIR.glob("run/*.yaml")):
            pipeline = Pipeline.from_mapping(compose([f"run={path.stem}"]))
            self.assertTrue(pipeline.steps, path)

    def test_a_dotted_override_reaches_the_environment(self):
        cfg = compose(["run=phase1b-1.7b", "run.env.MI_LAB_EVAL_SENTENCES=7"])
        self.assertEqual("7", Pipeline.from_mapping(cfg).env["MI_LAB_EVAL_SENTENCES"])

    def test_a_missing_directory_is_refused(self):
        with self.assertRaises(PipelineError):
            compose([], directory=Path("/nonexistent/pipelines"))


class TestRun(TestCase):
    """Steps are real subprocesses of this interpreter, so the modules are tiny scripts written here"""

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.addCleanup(set_log_file, None)
        # a module that appends its config to a file, and on its first call claims it hit its budget
        (self.root / "stepmod.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "log = Path(sys.argv[2]) / 'calls.txt'\n"
            "calls = log.read_text().splitlines() if log.exists() else []\n"
            "log.write_text('\\n'.join([*calls, sys.argv[1]]) + '\\n')\n"
            f"print('{BUDGET_MARKER} budget' if len(calls) == 0 else 'done')\n"
        )
        (self.root / "failmod.py").write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n")

    def pipeline(self, **overrides) -> Pipeline:
        steps = [Step(name="one", module="stepmod", args=[str(self.root)], repeat=True)]
        base = {"name": "demo", "config": "gpt2-small", "steps": steps,
                "env": {ENV_ROOT: str(self.root / "results"), "PYTHONPATH": str(self.root)}}
        return Pipeline(**{**base, **overrides})

    def test_a_dry_run_executes_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run(self.pipeline(dry_run=True))
        self.assertEqual({}, state["done"])
        self.assertFalse((self.root / "calls.txt").exists())

    def test_a_repeat_step_is_reinvoked_on_its_budget_and_then_recorded(self):
        with contextlib.redirect_stdout(io.StringIO()):
            state = run(self.pipeline())
        record = state["done"]["stepmod:one"]
        self.assertEqual(2, record["invocations"])
        self.assertEqual(1, record["budget_exits"])
        self.assertEqual(["gpt2-small", "gpt2-small"], (self.root / "calls.txt").read_text().split())
        self.assertEqual(state, json.loads((self.root / "results" / STATE_FILE).read_text()))

    def test_a_recorded_step_is_skipped_on_reentry_unless_resume_is_off(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.pipeline())
            run(self.pipeline())
        self.assertEqual(2, len((self.root / "calls.txt").read_text().split()))
        with contextlib.redirect_stdout(io.StringIO()):
            run(self.pipeline(resume=False))
        self.assertEqual(3, len((self.root / "calls.txt").read_text().split()))

    def test_a_failing_step_stops_the_pipeline_with_its_stderr(self):
        steps = [Step(name="bad", module="failmod"), Step(name="one", module="stepmod", args=[str(self.root)])]
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(PipelineError):
            run(self.pipeline(steps=steps))
        self.assertIn("boom", out.getvalue())
        self.assertFalse((self.root / "calls.txt").exists())
