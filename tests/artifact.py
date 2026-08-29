import json
import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from src.core.config import ModelConfig
from src.methods.probing import LinearProbe
from src.share import storage
from src.share.converters.activations import from_activations
from src.share.converters.probe import from_probe, to_probe
from src.share.converters.steering import from_steering
from src.share.loaders import open_probe
from src.share.schema.artifact import Artifact
from src.share.schema.control import Control
from src.share.schema.controls import Controls
from src.share.schema.errors import ArtifactError
from src.share.schema.metric import Metric
from src.share.schema.model import ModelRef
from src.share.schema.node import Node
from src.share.schema.payload import Payload
from src.share.schema.site import Site
from src.share.schema.span import Span
from src.share.schema.version import FORMAT, VERSION
from src.share.schema.vocabulary import Component, Kind, NodeComponent, Position
from src.share.storage import MANIFEST, TENSORS

"""
The artifact format is tested without a checkpoint, because everything that
can go wrong with it is structural. An artifact is wrong when its card and its
tensors disagree, when a grid's rows are a subset of layers that nothing wrote
down, when a recovery ships without the span it is a fraction of, when a
number ships without the definition that produced it, or when a probe comes
back as something that no longer scores. None of those need a
model to be wrong, and all of them are invisible in the numbers that come out.

The load-bearing test is the probe round trip. A format that cannot hand back
a working probe has not shared anything, whatever its manifest says.

The width used throughout is small and arbitrary. It is a shape, not a model
fact: these tests would pass identically on a residual stream of any size.
"""

WIDTH = 6
LAYERS = 4
HEADS = 3

def tiny_config() -> ModelConfig:
    """A resolved config for a model that does not exist, which is all the format needs"""
    return ModelConfig(
        id="tiny", backend="transformers", hf_name="nonexistent/tiny",
        n_layers=LAYERS, d_model=WIDTH, n_heads=HEADS,
    )

def tiny_circuit(**overrides) -> Artifact:
    """A minimal but complete circuit artifact"""
    fields = {
        "kind": "circuit",
        "id": "demo",
        "model": ModelRef.from_config(tiny_config()),
        "site": Site.at(range(LAYERS), LAYERS, component="head_out", position="all"),
        "span": Span(metric="logit_difference", clean=3.0, corrupted=0.5),
        "metrics": {"faithfulness": Metric(0.9, "recovery when only these heads are restored", "recovery")},
        "nodes": [Node(id="L2H1", component="head", layer=2, head=1, in_circuit=True, scores={"causal": 0.7})],
        "tensors": {
            "head_attribution": Payload(torch.zeros(LAYERS, HEADS), ["layer", "head"], "logits"),
            "head_effects": Payload(torch.zeros(LAYERS, HEADS), ["layer", "head"], "recovery"),
        },
    }
    fields.update(overrides)
    return Artifact(**fields)

def tiny_probe(**overrides) -> LinearProbe:
    """A probe whose numbers are distinctive enough that a round trip cannot fake them"""
    fields = {
        "weight": torch.linspace(-1, 1, WIDTH).double(),
        "bias": 0.25,
        "mean": torch.arange(WIDTH).double(),
        "std": torch.full((WIDTH,), 2.0, dtype=torch.float64),
        "layer": 2,
        "model_id": "tiny",
        "dataset": "cities",
        "method": "difference_of_means",
        "metrics": {"auc": 0.91},
    }
    fields.update(overrides)
    return LinearProbe(**fields)

class TestRoundTrip(TestCase):
    def test_a_circuit_survives_being_written_and_read(self):
        """Everything the card claims comes back, tensors included"""
        original = tiny_circuit()
        original.tensors["head_effects"].values.copy_(torch.randn(LAYERS, HEADS))
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save(original, str(Path(directory) / "demo.mia"))
            reloaded = storage.load(path)

        self.assertEqual(reloaded.kind, original.kind)
        self.assertEqual(reloaded.id, original.id)
        self.assertEqual(reloaded.model, original.model)
        self.assertEqual(reloaded.site, original.site)
        self.assertEqual(reloaded.span, original.span)
        self.assertEqual(reloaded.metrics, original.metrics)
        self.assertEqual(reloaded.controls, original.controls)
        self.assertEqual(reloaded.nodes, original.nodes)
        self.assertTrue(torch.allclose(reloaded.tensor("head_effects"), original.tensor("head_effects")))

    def test_axis_labels_come_back_with_their_tensor(self):
        """The token strings under a position axis are what make a heatmap redrawable"""
        tokens = ["Then", ",", " John", " and", " Mary"]
        artifact = tiny_circuit()
        artifact.tensors["residual_patch"] = Payload(
            torch.zeros(LAYERS, len(tokens)), ["layer", "position"], "recovery", {"position": tokens}
        )
        with tempfile.TemporaryDirectory() as directory:
            reloaded = storage.load(storage.save(artifact, str(Path(directory) / "demo.mia")))
        self.assertEqual(reloaded.tensors["residual_patch"].labels["position"], tokens)

    def test_the_card_is_readable_without_torch(self):
        """artifact.json is plain JSON, so deciding whether to download one costs nothing"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())

        self.assertEqual(manifest["format"], FORMAT)
        self.assertEqual(manifest["version"], VERSION)
        self.assertEqual(manifest["kind"], "circuit")
        self.assertEqual(manifest["tensors"]["head_effects"]["axes"], ["layer", "head"])
        self.assertEqual(manifest["measurement"]["span"]["clean"], 3.0)
        self.assertIn("definition", manifest["measurement"]["metrics"]["faithfulness"])

    def test_the_tensor_file_says_what_it_is_on_its_own(self):
        """A tensors.safetensors separated from its card still identifies itself"""
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            with safe_open(path / TENSORS, framework="pt") as handle:
                metadata = handle.metadata()
        self.assertEqual(metadata["format"], FORMAT)
        self.assertEqual(metadata["kind"], "circuit")

class TestValidation(TestCase):
    def test_a_kind_must_carry_what_that_kind_means(self):
        """A circuit without its causal grid is not an incomplete circuit, it is not one"""
        artifact = tiny_circuit()
        del artifact.tensors["head_effects"]
        with self.assertRaises(ArtifactError) as caught:
            artifact.validate()
        self.assertIn("head_effects", str(caught.exception))

    def test_a_recovery_cannot_ship_without_its_span(self):
        """Every fraction in a circuit divides by the span, so the span is not optional"""
        with self.assertRaises(ArtifactError) as caught:
            tiny_circuit(span=None).validate()
        self.assertIn("span", str(caught.exception))

    def test_a_tensor_must_name_one_axis_per_dimension(self):
        """A tensor stored without its axes is one the next reader transposes"""
        artifact = tiny_circuit()
        artifact.tensors["head_effects"] = Payload(torch.zeros(LAYERS, HEADS), ["layer"], "recovery")
        with self.assertRaises(ArtifactError) as caught:
            artifact.validate()
        self.assertIn("axis names", str(caught.exception))

    def test_a_grid_measured_over_a_subset_must_say_so(self):
        """Rows that are a subset of layers and are read as layer indices is the whole trap"""
        artifact = tiny_circuit(site=Site.at([0, 1], LAYERS, component="head_out"))
        with self.assertRaises(ArtifactError) as caught:
            artifact.validate()
        self.assertIn("layer", str(caught.exception))

    def test_labels_that_do_not_line_up_are_rejected(self):
        """Mislabelled ticks name the wrong column, silently"""
        artifact = tiny_circuit()
        artifact.tensors["head_effects"] = Payload(
            torch.zeros(LAYERS, HEADS), ["layer", "head"], "recovery", {"head": ["a", "b"]}
        )
        with self.assertRaises(ArtifactError) as caught:
            artifact.validate()
        self.assertIn("labels", str(caught.exception))

    def test_every_layer_carries_the_depth_it_sits_at(self):
        """Without the fraction, the site does not survive a model swap"""
        with self.assertRaises(ArtifactError) as caught:
            tiny_circuit(site=Site(layers=[0, 1, 2, 3], fracs=[], component="head_out")).validate()
        self.assertIn("fraction", str(caught.exception))

    def test_an_unknown_kind_is_an_error(self):
        with self.assertRaises(ArtifactError):
            tiny_circuit(kind="vibes").validate()

    def test_a_metric_cannot_ship_without_the_definition_that_produced_it(self):
        """'faithfulness' names two different quantities, so the number alone is not comparable"""
        artifact = tiny_circuit(metrics={"faithfulness": Metric(0.9, "  ", "recovery")})
        with self.assertRaises(ArtifactError) as caught:
            artifact.validate()
        self.assertIn("definition", str(caught.exception))

    def test_an_unknown_structural_assumption_is_refused(self):
        """Only assumptions that actually break the equivalence class may be claimed"""
        with self.assertRaises(ArtifactError) as caught:
            tiny_circuit(identifiability=["vibes"]).validate()
        self.assertIn("vibes", str(caught.exception))

class TestControls(TestCase):
    """Controls ship even when empty, for the reason edges does

    An artifact that ran no cross-task ablation and one that ran three have to
    be distinguishable. Absence that is written down is a finding; absence
    that is missing from the card reads as a question nobody thought to ask.
    """

    def test_an_empty_controls_block_is_still_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
        self.assertEqual(manifest["controls"], {"cross_task": [], "random_baseline": []})

    def test_a_cross_task_ablation_comes_back(self):
        """The check that a circuit is about its task and not shared infrastructure"""
        controls = Controls(cross_task=[
            Control(name="greater-than", metric="recovery", value=0.81, notes="ablating this circuit"),
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save(tiny_circuit(controls=controls), str(Path(directory) / "demo.mia"))
            reloaded = storage.load(path)
        self.assertEqual(reloaded.controls.cross_task[0].name, "greater-than")
        self.assertAlmostEqual(reloaded.controls.cross_task[0].value, 0.81)
        self.assertTrue(tiny_circuit().controls.empty)

class TestReading(TestCase):
    def test_a_card_with_unknown_keys_is_refused(self):
        """A key this reader does not know is a claim it would silently drop"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            manifest["confidence"] = "high"
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                storage.load(str(path))
        self.assertIn("confidence", str(caught.exception))

    def test_a_card_and_a_tensor_file_that_disagree_are_refused(self):
        """A tensor the card does not describe has no axes, so it cannot be read"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            del manifest["tensors"]["head_effects"]
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                storage.load(str(path))
        self.assertIn("head_effects", str(caught.exception))

    def test_a_future_version_is_refused_rather_than_guessed_at(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            manifest["version"] = "99.0"
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                storage.load(str(path))
        self.assertIn("99.0", str(caught.exception))

    def test_a_directory_that_is_not_an_artifact_says_so(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ArtifactError) as caught:
            storage.load(directory)
        self.assertIn(MANIFEST, str(caught.exception))

    def test_find_artifacts_skips_what_it_cannot_read(self):
        """One half-written artifact must not make a listing unusable"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage.save(tiny_circuit(), str(root / "good.mia"))
            (root / "broken.mia").mkdir()
            (root / "broken.mia" / MANIFEST).write_text("{not json")
            found = storage.find_artifacts(str(root))
        self.assertEqual([artifact.id for artifact in found], ["demo"])

class TestProbeArtifacts(TestCase):
    def test_a_probe_comes_back_as_a_working_probe(self):
        """The round trip is the test of the format: same scores, or nothing was shared"""
        probe = tiny_probe()
        activations = torch.randn(5, WIDTH, dtype=torch.float64)
        before = probe.score(activations)

        with tempfile.TemporaryDirectory() as directory:
            path = storage.save(from_probe(probe, cfg=tiny_config()), str(Path(directory) / "probe.mia"))
            restored = to_probe(storage.load(path))

        self.assertTrue(torch.allclose(before, restored.score(activations)))
        self.assertEqual(restored.layer, probe.layer)
        self.assertEqual(restored.model_id, probe.model_id)
        self.assertEqual(restored.method, probe.method)
        self.assertEqual(restored.dataset, probe.dataset)

    def test_the_steering_direction_travels_with_the_probe(self):
        """weight / std is the vector that steers, and a receiver must not have to derive it"""
        probe = tiny_probe()
        artifact = from_probe(probe, cfg=tiny_config())
        self.assertTrue(torch.allclose(artifact.tensor("direction").double(), probe.direction))

    def test_open_probe_takes_either_form(self):
        """Anything that only applies a probe should not care which file it was handed"""
        probe = tiny_probe()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe.save(str(root / "probe.pt"))
            storage.save(from_probe(probe, cfg=tiny_config()), str(root / "probe.mia"))
            from_file = open_probe(str(root / "probe.pt"))
            from_artifact = open_probe(str(root / "probe.mia"))
        self.assertTrue(torch.allclose(from_file.weight, from_artifact.weight))
        self.assertEqual(from_file.layer, from_artifact.layer)

    def test_a_probe_that_recorded_its_depth_packs_without_the_checkpoint(self):
        """The depth on the probe is what lets packing happen with no model in memory

        configs/ states no sizes -- they are read off checkpoints -- so a
        probe that did not write down its model's depth cannot be placed at
        one, and the fallback is the probe's own record.
        """
        artifact = from_probe(tiny_probe(model_id="gpt2-small", n_layers=12))
        self.assertEqual(artifact.site.layers, [2])
        self.assertAlmostEqual(artifact.site.fracs[0], 2 / 12, places=5)

    def test_a_probe_with_no_depth_anywhere_refuses_to_invent_one(self):
        """A site of bare indices is one the receiving lab cannot place"""
        with self.assertRaises(ValueError) as caught:
            from_probe(tiny_probe(model_id="gpt2-small"))
        self.assertIn("how many layers", str(caught.exception))

    def test_the_depth_survives_the_round_trip(self):
        probe = tiny_probe()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save(from_probe(probe, cfg=tiny_config()), str(Path(directory) / "probe.mia"))
            restored = to_probe(storage.load(path))
        self.assertEqual(restored.n_layers, LAYERS)
        self.assertAlmostEqual(restored.frac, probe.layer / LAYERS)

    def test_a_probe_whose_model_is_unknown_refuses_to_guess(self):
        """A ModelRef with an invented hf_name names a checkpoint that loads and differs"""
        with self.assertRaises(ValueError) as caught:
            from_probe(tiny_probe(model_id="not-a-config"))
        self.assertIn("not-a-config", str(caught.exception))

class TestOtherKinds(TestCase):
    def test_a_steering_vector_carries_its_sweep(self):
        """A direction without the curve that found its ceiling is untestable"""
        from src.methods.steering import SteeringPoint

        points = [SteeringPoint(strength=value, effect=value, fluency=1.0 - value / 4) for value in (0.0, 1.0, 2.0)]
        artifact = from_steering(
            tiny_config(), torch.ones(WIDTH), layer=2, source="difference_of_means", points=points,
        )
        self.assertEqual(artifact.kind, "steering_vector")
        self.assertEqual(list(artifact.tensor("strengths")), [0.0, 1.0, 2.0])
        self.assertEqual(artifact.site.layers, [2])
        self.assertAlmostEqual(artifact.site.fracs[0], 0.5)

    def test_an_activation_map_requires_its_axes(self):
        """A map whose columns are unnamed travels as a picture, which is the problem"""
        tokens = ["a", "b"]
        artifact = from_activations(
            tiny_config(), torch.zeros(LAYERS, len(tokens)), layers=range(LAYERS),
            axes=["layer", "position"], labels={"position": tokens},
        )
        self.assertEqual(artifact.tensors["values"].labels["position"], tokens)
        with self.assertRaises(ArtifactError):
            from_activations(tiny_config(), torch.zeros(LAYERS, 2), layers=range(LAYERS), axes=["layer"])

class TestProvenance(TestCase):
    def test_an_artifact_records_the_code_that_made_it(self):
        """A commit recorded from a dirty tree names code that never existed, so both are stored"""
        provenance = tiny_circuit().provenance
        self.assertEqual(provenance["tool"], "mi-lab")
        self.assertIn("git_dirty", provenance)
        self.assertIn("torch", provenance)

class TestVocabulary(TestCase):
    """The closed sets are closed, and the two `component` fields are not the same set

    A site says where a value was read from and a node says what part of the
    model it stands for. Before they were separate types nothing stopped one
    borrowing the other's names, and the card would round-trip either.
    """

    def test_a_site_cannot_take_a_node_component(self):
        with self.assertRaises(ArtifactError) as caught:
            Site.at([0], LAYERS, component="head")
        self.assertIn("head_out", str(caught.exception))

    def test_a_node_cannot_take_a_site_component(self):
        artifact = tiny_circuit(nodes=[Node(id="L2H1", component="head_out", layer=2, head=1)])
        with self.assertRaises(ArtifactError) as caught:
            artifact.validate()
        self.assertIn("L2H1", str(caught.exception))

    def test_a_mistyped_position_is_caught_where_it_is_written(self):
        """position was accepted unchecked before it had a type"""
        with self.assertRaises(ArtifactError) as caught:
            Site.at([0], LAYERS, position="lsat")
        self.assertIn("position", str(caught.exception))

    def test_a_card_naming_an_unknown_component_is_refused_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            manifest["site"]["component"] = "wherever"
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                storage.load(str(path))
        self.assertIn("wherever", str(caught.exception))

    def test_a_loaded_artifact_is_typed_the_same_as_a_built_one(self):
        """Half-coerced is its own bug: the same field should not be str here and a member there"""
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save(tiny_circuit(), str(Path(directory) / "demo.mia"))
            reloaded = storage.load(path)
        self.assertIs(reloaded.kind, Kind.CIRCUIT)
        self.assertIs(reloaded.site.component, Component.HEAD_OUT)
        self.assertIs(reloaded.site.position, Position.ALL)
        self.assertIs(reloaded.nodes[0].component, NodeComponent.HEAD)

    def test_a_member_serializes_as_its_plain_string(self):
        """The card is JSON a stranger reads, so an enum must not leak its class name"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(storage.save(tiny_circuit(), str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
        self.assertEqual(manifest["kind"], "circuit")
        self.assertEqual(manifest["site"]["component"], "head_out")
        self.assertEqual(manifest["graph"]["nodes"][0]["component"], "head")


class TestMigration(TestCase):
    """A version bump strands every artifact written before it, unless there is a way out

    v0.2 refuses a v0.1 card on purpose -- a bare float is the defect it fixes.
    That is only defensible with a migration beside it, and with a listing that
    says an artifact needs upgrading rather than quietly omitting it.
    """

    def _v1_card(self, directory: str) -> Path:
        """A v0.1 artifact: metrics as bare floats, no controls, no identifiability"""
        path = Path(directory) / "old.mia"
        storage.save(tiny_circuit(), str(path))
        manifest = json.loads((path / MANIFEST).read_text())
        manifest["version"] = "0.1"
        manifest["measurement"]["metrics"] = {"faithfulness": 0.919, "homegrown": 0.5}
        del manifest["measurement"]["identifiability"]
        del manifest["controls"]
        (path / MANIFEST).write_text(json.dumps(manifest))
        return path

    def test_an_old_card_names_the_command_that_fixes_it(self):
        """The reader used to leak a TypeError about a mapping from inside the build"""
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ArtifactError) as caught:
            storage.load(str(self._v1_card(directory)))
        message = str(caught.exception)
        self.assertIn("0.1", message)
        self.assertIn("artifact upgrade", message)

    def test_upgrading_recovers_the_definitions_this_lab_wrote(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._v1_card(directory)
            written, changes = storage.upgrade_in_place(str(path))
            upgraded = storage.load(written)

        self.assertEqual(upgraded.version, VERSION)
        self.assertIn("restored", upgraded.metrics["faithfulness"].definition)
        self.assertEqual(upgraded.metrics["faithfulness"].units, "recovery")
        self.assertTrue(any("faithfulness" in line for line in changes))

    def test_a_metric_it_does_not_know_says_the_definition_was_never_recorded(self):
        """Inventing a plausible sentence would defeat the gate it is passing"""
        with tempfile.TemporaryDirectory() as directory:
            written, _ = storage.upgrade_in_place(str(self._v1_card(directory)))
            upgraded = storage.load(written)
        self.assertIn("not recorded", upgraded.metrics["homegrown"].definition)

    def test_an_upgraded_card_says_it_was_upgraded(self):
        """A circuit that recorded no control must stay distinguishable from one that could not"""
        with tempfile.TemporaryDirectory() as directory:
            written, _ = storage.upgrade_in_place(str(self._v1_card(directory)))
            upgraded = storage.load(written)
        self.assertEqual(upgraded.provenance["upgraded_from"], "0.1")
        self.assertTrue(upgraded.controls.empty)

    def test_scan_reports_what_it_could_not_read_instead_of_dropping_it(self):
        with tempfile.TemporaryDirectory() as directory:
            self._v1_card(directory)
            storage.save(tiny_circuit(id="current"), str(Path(directory) / "new.mia"))
            found, problems = storage.scan(directory)
            # find_artifacts keeps its old shape for callers with nowhere to report;
            # inside the block, because the directory goes away with it
            readable = storage.find_artifacts(directory)

        self.assertEqual([artifact.id for artifact, _ in found], ["current"])
        self.assertEqual(len(problems), 1)
        self.assertIn("artifact upgrade", problems[0][1])
        self.assertEqual(len(readable), 1)

    def test_upgrading_something_current_says_there_is_nothing_to_do(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "new.mia")
            storage.save(tiny_circuit(), path)
            with self.assertRaises(ArtifactError) as caught:
                storage.upgrade_in_place(path)
        self.assertIn("already", str(caught.exception))
