"""Guards on the component vocabulary every knockout script speaks.

The names are trivial; the set algebra is where a study goes wrong quietly. A
band is a depth fraction resolved on the model at hand rather than a layer
range that only exists on one checkpoint, a head and its own layer's head
group are the same thing at two granularities, and the complement of a circuit
is what ablating "everything else" actually touches.
"""

from unittest import TestCase

from src.core.config import ModelConfig
from src.methods.components import (
    CANDIDATE_BAND,
    ComponentError,
    atomic_components,
    band,
    complement,
    head_components,
    heads_named,
    layer_components,
    mlps_named,
    name,
    parse,
    plan,
    redundant,
)


def depth(n_layers: int) -> ModelConfig:
    """A config of the given depth, so the band goes through the real `cfg.layer` rule"""
    return ModelConfig(id=f"depth-{n_layers}", backend="transformers", hf_name="none/tiny", n_layers=n_layers)


class TestNames(TestCase):
    def test_every_kind_round_trips(self):
        for cid in ("mlp:31", "heads:31", "head:31:7"):
            self.assertEqual(cid, name(*parse(cid)))

    def test_a_head_group_and_a_head_are_different_kinds(self):
        self.assertEqual(("heads", 3, None), parse("heads:3"))
        self.assertEqual(("head", 3, 5), parse("head:3:5"))

    def test_unknown_and_malformed_names_are_refused(self):
        for cid in ("neuron:3", "mlp", "head:3", "heads:3:1", "mlp:x"):
            with self.assertRaises(ComponentError):
                parse(cid)

    def test_a_single_head_needs_its_index(self):
        with self.assertRaises(ComponentError):
            name("head", 3)


class TestBand(TestCase):
    def test_the_candidate_band_is_the_top_quarter_on_any_depth(self):
        """The nine layers the study began on, and the proportional band elsewhere"""
        self.assertEqual(list(range(27, 36)), band(depth(36)))
        self.assertEqual([9, 10, 11], band(depth(12)))
        self.assertEqual((0.75, 1.0), CANDIDATE_BAND)

    def test_both_ends_are_inclusive(self):
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], band(depth(12), (0.0, 1.0)))


class TestSets(TestCase):
    def test_layer_components_are_mlps_then_head_groups(self):
        self.assertEqual(["mlp:1", "mlp:2", "heads:1", "heads:2"], layer_components([1, 2]))

    def test_head_components_walk_heads_within_layers(self):
        self.assertEqual(["head:4:0", "head:4:1", "head:5:0", "head:5:1"], head_components([4, 5], 2))

    def test_the_atomic_pool_never_holds_a_group(self):
        pool = atomic_components([4], 2)
        self.assertEqual(["mlp:4", "head:4:0", "head:4:1"], pool)
        self.assertNotIn("heads:4", pool)

    def test_the_upper_heads_run_deepest_first(self):
        self.assertEqual(["head:9:0", "head:8:0"], plan("heads-upper", [6, 7, 8, 9], 1))
        self.assertEqual(["head:6:0", "head:7:0"], plan("heads-lower", [6, 7, 8, 9], 1))
        self.assertEqual(layer_components([6, 7]), plan("layers", [6, 7], 1))

    def test_an_unknown_group_is_refused(self):
        with self.assertRaises(ComponentError):
            plan("neurons", [1], 1)

    def test_the_complement_is_every_other_whole_layer_component(self):
        rest = complement(["mlp:1", "heads:2"], 3)
        self.assertEqual(["mlp:0", "mlp:2", "heads:0", "heads:1"], rest)


class TestRedundancy(TestCase):
    def test_a_head_inside_a_chosen_group_is_redundant_both_ways(self):
        self.assertTrue(redundant("head:3:1", ["heads:3"]))
        self.assertTrue(redundant("heads:3", ["head:3:1"]))

    def test_the_mlp_of_the_same_layer_is_not(self):
        self.assertFalse(redundant("mlp:3", ["heads:3"]))
        self.assertFalse(redundant("head:3:1", ["head:4:1"]))

    def test_the_same_kind_twice_is_redundant(self):
        self.assertTrue(redundant("mlp:3", ["mlp:3"]))


class TestNamed(TestCase):
    def test_groups_expand_to_every_head(self):
        self.assertEqual({3: [0, 1], 5: [1]}, heads_named(["heads:3", "head:5:1", "mlp:2"], 2))

    def test_mlps_keep_first_mention_order_without_duplicates(self):
        self.assertEqual([9, 2], mlps_named(["mlp:9", "heads:9", "mlp:2", "mlp:9"]))
