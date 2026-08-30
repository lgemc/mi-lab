from unittest import TestCase

from src.data.tasks import build_task
from src.methods.circuits import CircuitError, ablate, behaviour, completeness
from src.methods.comparison import (
    compare_techniques,
    consistency,
    discover_across,
    specificity,
)
from src.methods.discovery import rank
from src.model.adapter import require_circuits
from src.share import storage
from src.share.converters.comparison import from_comparison

from .stubs.model import shared_adapter

"""
Comparison tests need a real checkpoint and skip loudly without one.

The load-bearing one is the last: an artifact written from a comparison has to
come back with its control slots *full*. Both slots have shipped empty since
the format existed, with a docstring saying that a circuit measured only on
its own task is not shown to be about that task -- and an artifact that ran a
cross-task ablation must not be byte-identical to one that never considered
it. The failure mode is quiet and was real: the cross-task sweep keys a task
by its registry name while an IOI dataset names itself after its corruption,
so the lookup missed and the controls came out empty, which reads exactly like
a check nobody ran.

The rest are the properties the three comparisons rest on. An agreement matrix
that is not symmetric is one whose rows and columns were swapped somewhere. An
ablation of nothing that is not a no-op means every damage number is a
difference against a moving baseline. And the empty subset in a completeness
sweep has to reproduce the circuit's own faithfulness, because that pair is
the anchor the rest of the curve is read against.
"""

LAYERS = [8, 9, 10, 11]

class ComparisonTestCase(TestCase):
    adapter = None
    task = None

    @classmethod
    def setUpClass(cls):
        adapter = shared_adapter()
        if adapter is None:
            raise cls.skipTest(cls, "gpt2-small is not available; run once with network access")
        cls.adapter = require_circuits(adapter)
        cls.task = build_task("ioi", cls.adapter, size=4, seed=0)

class TestAblation(ComparisonTestCase):
    def test_ablating_nothing_changes_nothing(self):
        """Every damage number is a difference against this being exactly zero"""
        empty = ablate(self.adapter, self.task, [])
        self.assertEqual(empty.damage, 0.0)
        self.assertEqual(empty.clean, empty.ablated)

    def test_damage_is_a_share_of_the_clean_behaviour(self):
        clean = behaviour(self.adapter, self.task)
        result = ablate(self.adapter, self.task, [(9, 9), (10, 7)])
        self.assertAlmostEqual(result.clean, clean.logit_difference, places=4)
        self.assertAlmostEqual(
            result.damage, (result.clean - result.ablated) / result.clean, places=6
        )

    def test_an_unknown_donor_names_the_ones_that_exist(self):
        with self.assertRaises(CircuitError) as raised:
            ablate(self.adapter, self.task, [(9, 9)], donor="zero")
        self.assertIn("mean", str(raised.exception))

class TestCompleteness(ComparisonTestCase):
    def test_the_empty_subset_is_the_circuit_against_an_untouched_model(self):
        circuit = rank("eap", self.adapter, self.task, layers=LAYERS).select(count=3)
        coverage = completeness(self.adapter, self.task, circuit, samples=3, seed=0)
        self.assertEqual(coverage.subsets[0], [])
        self.assertEqual(coverage.model_scores[0], 1.0)
        self.assertEqual(len(coverage.gaps), len(coverage.subsets))

    def test_an_empty_circuit_has_no_subsets(self):
        circuit = rank("eap", self.adapter, self.task, layers=LAYERS).select(count=1)
        circuit.heads.clear()
        with self.assertRaises(CircuitError):
            completeness(self.adapter, self.task, circuit)

class TestTechniqueComparison(ComparisonTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # the whole model, not the four-layer sweep the other classes use: ranking by
        # absolute score over the late layers alone selects mostly negative heads, and a
        # set of those restores *less* than a random one. That is a real property rather
        # than a flaw in the technique, and it is not the property this class is testing
        cls.comparison = compare_techniques(
            cls.adapter, cls.task, methods=["attribution", "patching", "eap", "random"],
            count=6, samples=2,
        )

    def test_every_technique_is_checked_the_same_way_at_the_same_size(self):
        for result in self.comparison.results:
            with self.subTest(technique=result.method):
                self.assertEqual(len(result.heads), 6)
                self.assertIsNotNone(result.report)
                self.assertIsNotNone(result.coverage)

    def test_the_agreement_matrices_are_symmetric_with_a_full_diagonal(self):
        for which in ("overlap", "order"):
            grid = self.comparison.matrix(which)
            with self.subTest(matrix=which):
                self.assertEqual(len(grid), len(self.comparison.methods))
                for row in range(len(grid)):
                    self.assertEqual(grid[row][row], 1.0)
                    for column in range(len(grid)):
                        self.assertAlmostEqual(grid[row][column], grid[column][row], places=9)

    def test_a_real_technique_beats_the_random_control(self):
        found = self.comparison.by_method()
        self.assertGreater(found["patching"].faithfulness, found["random"].faithfulness)

    def test_an_unknown_technique_is_refused_before_the_model_is_asked(self):
        with self.assertRaises(ValueError):
            compare_techniques(self.adapter, self.task, methods=["edge_patching"], check=False)

class TestConsistency(ComparisonTestCase):
    def test_it_finds_one_circuit_per_example(self):
        found = consistency(self.adapter, self.task, method="eap", count=3, examples=3, layers=LAYERS)
        self.assertEqual(len(found.circuits), 3)
        self.assertTrue(all(len(circuit) == 3 for circuit in found.circuits))
        self.assertGreaterEqual(found.reuse, 0.0)
        self.assertLessEqual(found.reuse, 1.0)

    def test_the_shared_set_is_what_appears_often_enough(self):
        found = consistency(self.adapter, self.task, method="eap", count=3, examples=4, layers=LAYERS)
        for head in found.shared:
            self.assertGreaterEqual(found.frequency[head], found.presence)

    def test_one_example_cannot_be_compared_with_itself(self):
        with self.assertRaises(CircuitError):
            consistency(self.adapter, self.task, examples=1)

class TestSpecificity(ComparisonTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tasks = {
            name: build_task(name, cls.adapter, size=4, seed=0) for name in ("ioi", "induction")
        }
        cls.circuits = discover_across(cls.adapter, cls.tasks, method="eap", count=4, layers=LAYERS)

    def test_every_circuit_is_ablated_on_every_task(self):
        found = specificity(self.adapter, self.tasks, self.circuits)
        self.assertEqual(len(found.matrix()), 2)
        for measured in found.tasks:
            self.assertAlmostEqual(found.matrix()[found.tasks.index(measured)][
                found.tasks.index(measured)], found.own(measured), places=9)
            self.assertIn(measured, found.control)

    def test_one_task_is_not_a_specificity_claim(self):
        with self.assertRaises(CircuitError):
            specificity(self.adapter, {"ioi": self.task}, {"ioi": self.circuits["ioi"]})

    def test_a_task_without_a_circuit_is_refused(self):
        with self.assertRaises(CircuitError):
            specificity(self.adapter, self.tasks, {"ioi": self.circuits["ioi"]})

class TestArtifact(ComparisonTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tasks = {
            name: build_task(name, cls.adapter, size=4, seed=0) for name in ("ioi", "induction")
        }
        circuits = discover_across(cls.adapter, cls.tasks, method="eap", count=3, layers=LAYERS)
        cls.across = specificity(cls.adapter, cls.tasks, circuits)
        cls.comparison = compare_techniques(
            cls.adapter, cls.tasks["ioi"], methods=["attribution", "patching", "random"],
            count=3, samples=2, layers=LAYERS,
        )
        cls.recurrence = consistency(
            cls.adapter, cls.tasks["ioi"], method="eap", count=3, examples=3, layers=LAYERS
        )

    def _artifact(self, **overrides):
        return from_comparison(
            self.adapter.cfg, self.tasks["ioi"], self.comparison,
            consistency=self.recurrence, specificity=self.across, task_key="ioi",
            **overrides,
        )

    def test_the_controls_come_back_full(self):
        """A check that was run and one that was never considered must not read the same"""
        artifact = self._artifact()
        self.assertFalse(artifact.controls.empty)
        self.assertTrue(artifact.controls.cross_task)
        self.assertTrue(artifact.controls.random_baseline)
        self.assertIn("induction", [control.name for control in artifact.controls.cross_task])

    def test_a_task_the_sweep_does_not_cover_is_refused_rather_than_skipped(self):
        with self.assertRaises(ValueError) as raised:
            from_comparison(
                self.adapter.cfg, self.tasks["ioi"], self.comparison, specificity=self.across,
                task_key="greater_than",
            )
        self.assertIn("cross-task", str(raised.exception))

    def test_every_technique_keeps_its_own_units(self):
        artifact = self._artifact()
        self.assertEqual(artifact.tensors["head_attribution"].units, "logits")
        self.assertEqual(artifact.tensors["head_effects"].units, "recovery")
        self.assertIn("scores_random", artifact.tensors)

    def test_a_comparison_without_the_measured_grids_cannot_be_a_circuit(self):
        cheap = compare_techniques(
            self.adapter, self.tasks["ioi"], methods=["eap", "random"], count=3, check=False, layers=LAYERS
        )
        with self.assertRaises(ValueError) as raised:
            from_comparison(self.adapter.cfg, self.tasks["ioi"], cheap, reference="eap")
        self.assertIn("head_attribution", str(raised.exception))

    def test_it_round_trips_through_storage(self):
        import tempfile
        from pathlib import Path

        artifact = self._artifact()
        with tempfile.TemporaryDirectory() as directory:
            path = storage.save(artifact, str(Path(directory) / f"comparison{storage.SUFFIX}"))
            loaded = storage.load(path)
        self.assertEqual(loaded.id, artifact.id)
        self.assertEqual(len(loaded.controls.cross_task), len(artifact.controls.cross_task))
        self.assertEqual(sorted(loaded.tensors), sorted(artifact.tensors))
        for metric in loaded.metrics.values():
            self.assertTrue(metric.definition.strip())
