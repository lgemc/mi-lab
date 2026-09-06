"""The circuit server: circuits swapped in and out of one resident model, over HTTP

Driven through FastAPI's test client on GPT-2 small, with circuits written the
way `scripts/sheaf_prune.py` writes them -- both kinds, because the server
picks how to run one by reading its folder and the two are not the same
object. The properties that matter:

  the restore      after any request, under any circuit, the weights are what
                   they were. A weight mask multiplies into them; the failure
                   is the *next* request answering through half a circuit.
  the identity     a circuit that closes nothing must reproduce the full model
                   exactly, for either kind. Asserted on generated text, which
                   is what a caller actually gets.
  the folder       nothing is read until a request names it, a folder written
                   after startup is served without a restart, and a circuit
                   that cannot run here is listed saying why rather than
                   silently missing.
"""

import json
import tempfile
import time
import unittest
from pathlib import Path

import torch

from src.methods.gates import MASK_FILE, pack
from src.methods.sheaves import gateable
from src.model.adapter import require_circuits

from .stubs.model import shared_adapter

try:
    from fastapi.testclient import TestClient

    from src.serve.app import create_app
    from src.serve.circuits import FULL, Circuits, ServeError
    from src.serve.examples import Examples
    from src.serve.models import ModelPool, ModelSpec
except ImportError:  # the serve extra is optional
    TestClient = None


@unittest.skipIf(TestClient is None, "serve extra not installed (uv sync --extra serve)")
class TestServe(unittest.TestCase):
    adapter = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise unittest.SkipTest("gpt2-small is not available")
        cls.adapter = require_circuits(adapter)
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        targets = gateable(cls.adapter)
        generator = torch.Generator().manual_seed(0)
        # a half-open random mask over every gateable tensor, in a nested
        # directory so the served name is the relative path
        gates = {name: torch.rand(p.shape, generator=generator) - 0.5 for name, p in targets.items()}
        (root / "sweep" / "half").mkdir(parents=True)
        torch.save(pack(gates), root / "sweep" / "half" / MASK_FILE.format(task="ioi"))
        (root / "sweep" / "half" / "sheaf-ioi.json").write_text(json.dumps({"accuracy": 0.5}))
        # an all-open mask: must reproduce the full model exactly
        (root / "open").mkdir()
        torch.save(pack({name: torch.ones(p.shape) for name, p in targets.items()}),
                   root / "open" / MASK_FILE.format(task="ioi"))
        # an edge circuit with every edge open: the other kind of folder, and
        # the same identity claim -- it must reproduce the full model exactly
        (root / "edges" / "all-open").mkdir(parents=True)
        (root / "edges" / "all-open" / "sheaf-ioi.json").write_text(json.dumps({
            "config": "gpt2-small", "task": "ioi",
            "edges_open": [list(edge) for edge in cls.adapter.edges()],
            "n_edges": len(cls.adapter.edges()), "edge_density": 1.0, "accuracy": 0.9,
        }))
        cls.root = root
        cls.circuits = Circuits(cls.adapter, root, tasks=["ioi"])
        cls.client = TestClient(create_app(cls.circuits, config="gpt2-small"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_discovery_names_and_reports(self):
        self.assertEqual(self.circuits.names(),
                         ["full", "edges/all-open", "open", "sweep/half"])
        half = self.circuits.get("sweep/half")
        self.assertAlmostEqual(half.density, 0.5, delta=0.01)
        self.assertEqual(half.spec.report["held_out_ranking"], 0.5)
        self.assertEqual(self.circuits.get("open").density, 1.0)
        with self.assertRaises(ServeError):
            self.circuits.get("missing")

    def test_each_folder_is_claimed_by_the_backbone_that_can_run_it(self):
        """The registry's whole job: a mask folder and an edge folder are different objects"""
        self.assertEqual(self.circuits.specs["sweep/half"].kind, "weights")
        self.assertEqual(self.circuits.specs["edges/all-open"].kind, "edges")
        self.assertEqual(self.circuits.get("sweep/half").unit, "weights")
        self.assertEqual(self.circuits.get("edges/all-open").unit, "edges")

    def test_an_edge_circuit_with_every_edge_open_is_the_full_model(self):
        """The identity for the other kind. It was invisible to the server before this"""
        response = self.client.post("/infer", json={
            "inputs": ["The capital of France is"], "circuits": ["full", "edges/all-open"],
            "max_new_tokens": 4})
        self.assertEqual(response.status_code, 200, response.text)
        outputs = response.json()["outputs"]
        self.assertEqual(outputs["full"], outputs["edges/all-open"])

    def test_health_and_listing(self):
        health = self.client.get("/health").json()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["config"], "gpt2-small")
        self.assertEqual(health["tasks"], ["ioi"])
        self.assertEqual(health["circuits"], ["full", "edges/all-open", "open", "sweep/half"])
        # The frames are keyed by task and carry both holes, because the page
        # draws the prompt from them: one hard-coded template meant an IOI
        # server showed a translation frame and framed every input as a
        # Spanish word. A task with no one-input frame has no entry, which is
        # how the page knows to take the whole prompt instead.
        self.assertEqual(health["frames"]["translation"], "Spanish: {input}\nEnglish:{answer}")
        self.assertNotIn("ioi", health["frames"])
        self.assertEqual(sorted(health["frames"]), health["framed_tasks"])
        listed = self.client.get("/circuits").json()
        self.assertEqual([c["name"] for c in listed["circuits"]],
                         ["edges/all-open", "open", "sweep/half"])
        self.assertIn("density", listed["circuits"][0])
        self.assertEqual({c["kind"] for c in listed["circuits"]}, {"weights", "edges"})


    def test_generate_under_each_circuit_and_restore(self):
        before = {name: p.detach().clone() for name, p in gateable(self.adapter).items()}
        response = self.client.post("/infer", json={
            "inputs": ["The capital of France is"], "circuits": ["full", "open", "sweep/half"],
            "max_new_tokens": 3})
        self.assertEqual(response.status_code, 200, response.text)
        outputs = response.json()["outputs"]
        self.assertEqual(set(outputs), {"full", "open", "sweep/half"})
        self.assertEqual(outputs["full"], outputs["open"])
        for name, p in gateable(self.adapter).items():
            self.assertTrue(torch.equal(p.detach(), before[name]), name)

    def test_a_named_task_frames_the_input_and_keeps_the_first_line(self):
        response = self.client.post("/infer", json={
            "inputs": ["perro"], "task": "translation", "circuits": ["full"]})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["inputs"], ["perro"])
        self.assertEqual(body["task"], "translation")
        # the prompt comes back, because a caller who cannot see it cannot tell
        # a bad circuit from a badly framed question
        self.assertEqual(body["prompts"], ["Spanish: perro\nEnglish:"])
        self.assertEqual(len(body["outputs"]["full"]), 1)
        self.assertNotIn("\n", body["outputs"]["full"][0])

    def test_no_task_sends_whole_prompts_untrimmed(self):
        response = self.client.post("/infer", json={
            "inputs": ["The capital of France is"], "circuits": ["full"], "max_new_tokens": 4})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNone(body["task"])
        self.assertEqual(body["prompts"], body["inputs"])

    def test_a_task_with_no_one_input_frame_says_so(self):
        """IOI prompts are whole sentences; there is no single hole to fill"""
        response = self.client.post("/infer", json={
            "inputs": ["Jack"], "task": "ioi", "circuits": ["full"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("one-input frame", response.json()["detail"])

    def test_unknown_circuit_is_a_bad_request(self):
        response = self.client.post("/infer", json={"inputs": ["x"], "circuits": ["nope"]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("nope", response.json()["detail"])
        self.assertEqual(self.client.post("/infer", json={"inputs": []}).status_code, 422)

    def test_page(self):
        """The page is served, and it is the page: a model picker over the routes above it"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("mi-lab", response.text)
        # asserted rather than the title, which is prose: what makes the page
        # the page is that it offers the circuits and can reach the routes
        for marker in ('id="model"', 'id="checkpoint"', "`circuits?model=", "fetch('infer'"):
            self.assertIn(marker, response.text, marker)


class TestFolderIsTheDeployment(unittest.TestCase):
    """The model is deployed; the circuits are a folder read at request time

    Each of these was a way the previous server made publishing a circuit mean
    rebuilding an image, or made a circuit it could not run indistinguishable
    from one that was never written.
    """

    adapter = None

    @classmethod
    def setUpClass(cls):
        if TestClient is None:
            raise unittest.SkipTest("serve extra not installed")
        adapter = shared_adapter()
        if adapter is None:
            raise unittest.SkipTest("gpt2-small is not available")
        cls.adapter = require_circuits(adapter)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write_edges(self, name, *, config="gpt2-small", fraction=1.0):
        every = list(self.adapter.edges())
        kept = every[:max(1, int(len(every) * fraction))]
        directory = self.root / name
        directory.mkdir(parents=True)
        (directory / "sheaf-ioi.json").write_text(json.dumps({
            "config": config, "task": "ioi", "edges_open": [list(e) for e in kept],
            "n_edges": len(every), "edge_density": len(kept) / len(every),
        }))
        return directory

    def test_nothing_is_read_until_a_request_names_it(self):
        """Startup scans; it does not load. The probe used to wait out every mask"""
        self.write_edges("a")
        self.write_edges("b")
        circuits = Circuits(self.adapter, self.root, tasks=["ioi"])
        self.assertEqual(sorted(circuits.specs), ["a", "b"])
        self.assertEqual(circuits.resident, {})
        circuits.get("a")
        self.assertEqual(sorted(circuits.resident), ["a"])

    def test_a_folder_written_after_startup_is_served(self):
        """The whole point: publishing a circuit is writing a folder, not a redeploy"""
        self.write_edges("first")
        circuits = Circuits(self.adapter, self.root, tasks=["ioi"])
        self.assertEqual(circuits.names(), ["full", "first"])
        self.write_edges("later")
        # named directly, without listing first -- `get` rescans on a miss
        self.assertIsNotNone(circuits.get("later"))
        self.assertIn("later", circuits.names())

    def test_a_circuit_for_another_model_is_listed_with_its_reason(self):
        """Refused, but visibly. A circuit that is silently absent cannot be chased"""
        self.write_edges("foreign", config="some-other-model")
        circuits = Circuits(self.adapter, self.root, tasks=["ioi"])
        self.assertIn("foreign", circuits.specs)
        self.assertNotIn("foreign", circuits.names())
        self.assertIn("some-other-model", circuits.specs["foreign"].problem)
        with self.assertRaises(ServeError) as caught:
            circuits.get("foreign")
        self.assertIn("some-other-model", str(caught.exception))

    def test_a_folder_nothing_claims_is_reported_not_dropped(self):
        (self.root / "empty").mkdir()
        (self.root / "empty" / "sheaf-ioi.json").write_text(json.dumps({"config": "gpt2-small"}))
        circuits = Circuits(self.adapter, self.root, tasks=["ioi"])
        # the task is in the label: with several tasks under one root, "empty"
        # alone does not say which circuit is missing
        self.assertEqual(circuits.skipped, ["empty [ioi]"])
        self.assertEqual(circuits.names(), ["full"])

    def test_residency_is_capped_and_the_evicted_one_still_works(self):
        """Evicting is free: the folder is still there and the next request rereads it"""
        self.write_edges("a")
        self.write_edges("b")
        circuits = Circuits(self.adapter, self.root, tasks=["ioi"], max_resident=1)
        circuits.get("a")
        circuits.get("b")
        self.assertEqual(sorted(circuits.resident), ["b"])
        self.assertIsNotNone(circuits.get("a"))
        self.assertEqual(sorted(circuits.resident), ["a"])


if __name__ == "__main__":
    unittest.main()


class TestManyTasks(unittest.TestCase):
    """One model, one process, every task the folder holds

    Pinning the server to a single task meant an IOI circuit and a translation
    circuit pruned on the same checkpoint needed two deployments of the same
    7 GiB of weights. The task is a property of the folder, so the server reads
    it rather than being told.
    """

    adapter = None

    @classmethod
    def setUpClass(cls):
        if TestClient is None:
            raise unittest.SkipTest("serve extra not installed")
        adapter = shared_adapter()
        if adapter is None:
            raise unittest.SkipTest("gpt2-small is not available")
        cls.adapter = require_circuits(adapter)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, task, fraction=1.0):
        every = list(self.adapter.edges())
        kept = every[:max(1, int(len(every) * fraction))]
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"sheaf-{task}.json").write_text(json.dumps({
            "config": "gpt2-small", "task": task,
            "edges_open": [list(e) for e in kept],
            "n_edges": len(every), "edge_density": len(kept) / len(every),
        }))
        return directory

    def test_both_tasks_are_discovered_without_being_named(self):
        self.write("a-ioi", "ioi")
        self.write("b-translation", "translation")
        circuits = Circuits(self.adapter, self.root)
        self.assertEqual(circuits.tasks, ["ioi", "translation"])
        self.assertEqual(circuits.names(), ["full", "a-ioi", "b-translation"])
        self.assertEqual(circuits.specs["a-ioi"].task, "ioi")
        self.assertEqual(circuits.specs["b-translation"].task, "translation")

    def test_naming_only_moves_for_the_directory_that_collides(self):
        """A second task appearing must not rename a circuit somewhere else"""
        self.write("both", "ioi")
        self.write("both", "translation")
        self.write("alone", "ioi")
        circuits = Circuits(self.adapter, self.root)
        self.assertIn("both:ioi", circuits.specs)
        self.assertIn("both:translation", circuits.specs)
        self.assertIn("alone", circuits.specs)
        self.assertNotIn("alone:ioi", circuits.specs)

    def test_serving_one_task_ignores_the_others(self):
        self.write("a-ioi", "ioi")
        self.write("b-translation", "translation")
        circuits = Circuits(self.adapter, self.root, tasks=["ioi"])
        self.assertEqual(circuits.tasks, ["ioi"])
        self.assertEqual(circuits.names(), ["full", "a-ioi"])

    def test_an_unknown_task_is_refused_at_construction(self):
        with self.assertRaises(ServeError) as caught:
            Circuits(self.adapter, self.root, tasks=["not-a-task"])
        self.assertIn("not-a-task", str(caught.exception))

    def test_a_sibling_artifact_is_not_read_as_a_task(self):
        """`sheaf-translation-inference.json` is not a task called translation-inference"""
        directory = self.write("run", "translation")
        (directory / "sheaf-translation-inference.json").write_text(json.dumps({"circuit": {}}))
        (directory / "sheaf-translation-circuit.json").write_text(json.dumps({"components": []}))
        circuits = Circuits(self.adapter, self.root)
        self.assertEqual(circuits.tasks, ["translation"])
        self.assertEqual(circuits.names(), ["full", "run"])

    def test_translate_refuses_a_circuit_pruned_for_another_task(self):
        """Framing an IOI circuit as translation returns something, and it means nothing"""
        self.write("ioi-run", "ioi")
        circuits = Circuits(self.adapter, self.root)
        client = TestClient(create_app(circuits, config="gpt2-small"))
        response = client.post("/infer", json={"inputs": ["perro"], "task": "translation",
                                              "circuits": ["ioi-run"]})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("ioi", response.json()["detail"])
        # the same door with no task takes it: those are whole prompts
        ok = client.post("/infer", json={"inputs": ["Then, Jack and Sam went"],
                                         "circuits": ["ioi-run"], "max_new_tokens": 2})
        self.assertEqual(ok.status_code, 200, ok.text)


@unittest.skipIf(TestClient is None, "serve extra not installed (uv sync --extra serve)")
class TestGenerationServer(unittest.TestCase):
    """A checkpoint served for its own answers, with no circuits over it

    This is the SDFT stack's shape (`configs/qwen3-0.6b-sdft.yaml`). The claims
    that matter are that it costs nothing it does not use -- no clean-weight
    copy, which is the whole second model in memory -- and that a *mistyped*
    circuits path still fails loudly instead of quietly becoming this mode.
    """

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise unittest.SkipTest("gpt2-small is not available")
        cls.adapter = require_circuits(adapter)
        cls.circuits = Circuits(cls.adapter, None)
        cls.client = TestClient(create_app(cls.circuits, config="gpt2-small"))

    def test_no_circuits_means_no_clean_weight_copy(self):
        """The copy exists to undo a mask; with no mask it is a duplicate model for nothing"""
        self.assertEqual(self.circuits.host.originals, {})
        self.assertEqual(self.circuits.host.targets, {})

    def test_only_the_full_model_is_served(self):
        self.assertEqual(self.circuits.names(), [FULL])
        listing = self.client.get("/circuits").json()
        self.assertEqual(listing["circuits"], [])
        self.assertIsNone(listing["root"])

    def test_health_reports_no_root_rather_than_a_made_up_one(self):
        health = self.client.get("/health").json()
        self.assertIsNone(health["root"])
        self.assertEqual(health["circuits"], [FULL])

    def test_whole_prompts_are_generated(self):
        response = self.client.post("/infer", json={"inputs": ["The capital of France is"],
                                                    "max_new_tokens": 3})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["outputs"][FULL]), 1)

    def test_a_rescan_of_nothing_is_not_an_error(self):
        """/circuits rescans on every call, and there is nothing under None to walk"""
        self.assertEqual(self.circuits.scan(), {})
        self.assertEqual(self.client.get("/circuits").status_code, 200)


class TestChatWrapping(unittest.TestCase):
    """An instruct checkpoint is asked a conversation; a base one is not

    Driven off the config rather than the checkpoint name, so the wrapping is a
    decision written down in `configs/` and every existing base-model prompt
    stays byte-identical.
    """

    @classmethod
    def setUpClass(cls):
        cls.adapter = shared_adapter()
        if cls.adapter is None:
            raise unittest.SkipTest("gpt2-small is not available")

    def test_a_base_model_prompt_is_untouched(self):
        self.assertFalse(self.adapter.cfg.chat)
        self.assertEqual(self.adapter.chat_wrap(["hello"]), ["hello"])

    def test_chat_true_puts_the_prompt_through_the_template(self):
        if self.adapter.tokenizer.chat_template is None:
            self.skipTest("gpt2-small ships no chat template")
        self.adapter.cfg.chat = True
        try:
            self.assertNotEqual(self.adapter.chat_wrap(["hello"]), ["hello"])
        finally:
            self.adapter.cfg.chat = False


@unittest.skipIf(TestClient is None, "serve extra not installed (uv sync --extra serve)")
class TestExamples(unittest.TestCase):
    """Held-out prompts as data under a root, read per request"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_missing_root_is_empty_not_an_error(self):
        """The circuit stacks mount no examples, and that is a normal server"""
        loaded = Examples(self.root / "nope").load()
        self.assertEqual(loaded["datasets"], [])
        self.assertEqual(loaded["problems"], [])

    def test_a_file_written_after_construction_is_served(self):
        examples = Examples(self.root)
        self.assertEqual(examples.load()["datasets"], [])
        (self.root / "tooluse.json").write_text(json.dumps({
            "dataset": "tooluse", "split": "eval", "split_size": 97,
            "examples": [{"index": 0, "prompt": "p", "gold": []}],
        }))
        [dataset] = examples.load()["datasets"]
        self.assertEqual((dataset["dataset"], dataset["count"], dataset["split"]),
                         ("tooluse", 1, "eval"))

    def test_a_broken_file_is_named_rather_than_dropped(self):
        """A dataset silently absent looks exactly like one nobody exported"""
        (self.root / "science.json").write_text("{not json")
        (self.root / "tooluse.json").write_text(json.dumps({"dataset": "tooluse"}))
        loaded = Examples(self.root).load()
        self.assertEqual(loaded["datasets"], [])
        self.assertEqual(len(loaded["problems"]), 2)
        self.assertTrue(any("science.json" in p for p in loaded["problems"]))
        self.assertTrue(any("no 'examples' list" in p for p in loaded["problems"]))


@unittest.skipIf(TestClient is None, "serve extra not installed (uv sync --extra serve)")
class TestPooledApp(unittest.TestCase):
    """Two checkpoints behind one app, named per request and unloaded when idle

    Driven with a fake builder: what is under test here is the routing and the
    busy-marking, not the loading, and the loading has its own module
    (`tests/pool.py`). The claim that matters is the one the GPU cares about --
    an in-flight generation is never swept.
    """

    def setUp(self):
        adapter = shared_adapter()
        if adapter is None:
            self.skipTest("gpt2-small is not available")
        self.adapter = require_circuits(adapter)
        self.built = []
        specs = [ModelSpec("first", "gpt2-small"), ModelSpec("second", "gpt2-small")]
        self.pool = ModelPool(specs, idle_timeout=0.05, build=self.build)
        self.client = TestClient(create_app(pool=self.pool))

    def build(self, spec):
        self.built.append(spec.name)
        return Circuits(self.adapter, None, config=spec.config)

    def test_health_loads_nothing(self):
        """The probe runs forever; a probe that loads a checkpoint is a restart loop"""
        health = self.client.get("/health").json()
        self.assertEqual(self.built, [])
        self.assertEqual(health["resident_models"], [])
        self.assertEqual([m["name"] for m in health["models"]], ["first", "second"])
        # and it still reports the budget, because that comes from the config
        self.assertGreater(health["max_new_tokens"], 0)

    def test_the_default_model_is_the_first(self):
        self.assertEqual(self.client.get("/health").json()["model"], "first")

    def test_an_unknown_model_is_a_bad_request_naming_the_real_ones(self):
        response = self.client.get("/health?model=nope")
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("first", response.json()["detail"])

    def test_infer_loads_the_named_model_and_says_which(self):
        body = {"inputs": ["The capital of France is"], "max_new_tokens": 2, "model": "second"}
        result = self.client.post("/infer", json=body).json()
        self.assertEqual(result["model"], "second")
        self.assertEqual(self.built, ["second"])

    def test_an_idle_model_is_dropped_and_reloaded_on_the_next_request(self):
        body = {"inputs": ["hello"], "max_new_tokens": 1}
        self.client.post("/infer", json=body)
        time.sleep(0.08)
        self.assertEqual(self.pool.sweep(), ["first"])
        self.client.post("/infer", json=body)
        self.assertEqual(self.built, ["first", "first"])

    def test_health_does_not_pin_the_default_model(self):
        """The probe runs every ten seconds; if it counted as a use, nothing would sweep"""
        self.client.post("/infer", json={"inputs": ["hello"], "max_new_tokens": 1})
        time.sleep(0.04)
        self.client.get("/health")
        time.sleep(0.02)
        self.assertEqual(self.pool.sweep(), ["first"])
