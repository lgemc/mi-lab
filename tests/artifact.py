import json
import tempfile
from pathlib import Path
from unittest import TestCase

import torch

from src.core.config import ModelConfig
from src.methods.probing import LinearProbe
from src.share.artifact import (
    FORMAT,
    MANIFEST,
    TENSORS,
    VERSION,
    Artifact,
    ArtifactError,
    ModelRef,
    Node,
    Payload,
    Site,
    Span,
    find_artifacts,
)
from src.share.sharing import from_activations, from_probe, from_steering, open_probe, to_probe

"""
The artifact format is tested without a checkpoint, because everything that
can go wrong with it is structural. An artifact is wrong when its card and its
tensors disagree, when a grid's rows are a subset of layers that nothing wrote
down, when a recovery ships without the span it is a fraction of, or when a
probe comes back as something that no longer scores. None of those need a
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
            path = original.save(str(Path(directory) / "demo.mia"))
            reloaded = Artifact.load(path)

        self.assertEqual(reloaded.kind, original.kind)
        self.assertEqual(reloaded.id, original.id)
        self.assertEqual(reloaded.model, original.model)
        self.assertEqual(reloaded.site, original.site)
        self.assertEqual(reloaded.span, original.span)
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
            reloaded = Artifact.load(artifact.save(str(Path(directory) / "demo.mia")))
        self.assertEqual(reloaded.tensors["residual_patch"].labels["position"], tokens)

    def test_the_card_is_readable_without_torch(self):
        """artifact.json is plain JSON, so deciding whether to download one costs nothing"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(tiny_circuit().save(str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())

        self.assertEqual(manifest["format"], FORMAT)
        self.assertEqual(manifest["version"], VERSION)
        self.assertEqual(manifest["kind"], "circuit")
        self.assertEqual(manifest["tensors"]["head_effects"]["axes"], ["layer", "head"])
        self.assertEqual(manifest["measurement"]["span"]["clean"], 3.0)

    def test_the_tensor_file_says_what_it_is_on_its_own(self):
        """A tensors.safetensors separated from its card still identifies itself"""
        from safetensors import safe_open

        with tempfile.TemporaryDirectory() as directory:
            path = Path(tiny_circuit().save(str(Path(directory) / "demo.mia")))
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

class TestReading(TestCase):
    def test_a_card_with_unknown_keys_is_refused(self):
        """A key this reader does not know is a claim it would silently drop"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(tiny_circuit().save(str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            manifest["confidence"] = "high"
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                Artifact.load(str(path))
        self.assertIn("confidence", str(caught.exception))

    def test_a_card_and_a_tensor_file_that_disagree_are_refused(self):
        """A tensor the card does not describe has no axes, so it cannot be read"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(tiny_circuit().save(str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            del manifest["tensors"]["head_effects"]
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                Artifact.load(str(path))
        self.assertIn("head_effects", str(caught.exception))

    def test_a_future_version_is_refused_rather_than_guessed_at(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(tiny_circuit().save(str(Path(directory) / "demo.mia")))
            manifest = json.loads((path / MANIFEST).read_text())
            manifest["version"] = "99.0"
            (path / MANIFEST).write_text(json.dumps(manifest))
            with self.assertRaises(ArtifactError) as caught:
                Artifact.load(str(path))
        self.assertIn("99.0", str(caught.exception))

    def test_a_directory_that_is_not_an_artifact_says_so(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ArtifactError) as caught:
            Artifact.load(directory)
        self.assertIn(MANIFEST, str(caught.exception))

    def test_find_artifacts_skips_what_it_cannot_read(self):
        """One half-written artifact must not make a listing unusable"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_circuit().save(str(root / "good.mia"))
            (root / "broken.mia").mkdir()
            (root / "broken.mia" / MANIFEST).write_text("{not json")
            found = find_artifacts(str(root))
        self.assertEqual([artifact.id for artifact in found], ["demo"])

class TestProbeArtifacts(TestCase):
    def test_a_probe_comes_back_as_a_working_probe(self):
        """The round trip is the test of the format: same scores, or nothing was shared"""
        probe = tiny_probe()
        activations = torch.randn(5, WIDTH, dtype=torch.float64)
        before = probe.score(activations)

        with tempfile.TemporaryDirectory() as directory:
            path = from_probe(probe, cfg=tiny_config()).save(str(Path(directory) / "probe.mia"))
            restored = to_probe(Artifact.load(path))

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
            from_probe(probe, cfg=tiny_config()).save(str(root / "probe.mia"))
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
            path = from_probe(probe, cfg=tiny_config()).save(str(Path(directory) / "probe.mia"))
            restored = to_probe(Artifact.load(path))
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
