"""The weight diff, on tensors whose answers are known in advance.

All of this is offline. A weight diff needs two checkpoints and nothing else --
no tokenizer, no forward pass, no accelerator -- so the fixtures here are
safetensors files written into a temporary directory, a few tokens wide, and the
measurements are checked against deltas constructed to have a particular rank.

The three that matter: a rank-one delta must report a stable rank of one and all
of its energy in the leading direction, an isotropic delta must report the full
rank of its shape, and a checkpoint that ships a duplicate of a tensor it
already has must be counted once and told about.
"""

import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from safetensors.torch import save_file

from src.domains.lm.analysis.diffing import (
    ATTENTION,
    FEEDFORWARD,
    TOP_K,
    DeltaReport,
    DiffingError,
    FitReport,
    LayerFit,
    checkpoint_delta,
    classify,
    duplicates,
    fit_shift,
    index,
    read,
    shapes,
    table,
    tensor_delta,
    unexplained,
)

ROWS, COLUMNS = 12, 8


def layer_tensors(index_of_layer: int, scale: float = 0.0, generator=None) -> dict:
    """One transformer layer's worth of named tensors, optionally nudged by `scale`

    Named exactly as a checkpoint names them, because `classify` reads the name
    and a test on synthetic tensors with invented names would check nothing.
    """
    out = {}
    for kind in ATTENTION + FEEDFORWARD:
        base = torch.eye(ROWS, COLUMNS)
        if scale:
            base = base + scale * torch.randn(ROWS, COLUMNS, generator=generator)
        out[f"model.layers.{index_of_layer}.{'self_attn' if kind in ATTENTION else 'mlp'}.{kind}.weight"] = base
    out[f"model.layers.{index_of_layer}.input_layernorm.weight"] = torch.ones(ROWS)
    return out


def checkpoint(directory: Path, tensors: dict, shards: int = 1) -> Path:
    """Write tensors as one or several safetensors files under `directory`"""
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    per = max(1, len(names) // shards + (1 if len(names) % shards else 0))
    for shard in range(shards):
        chunk = names[shard * per : (shard + 1) * per]
        if chunk:
            save_file({name: tensors[name] for name in chunk}, directory / f"model-{shard}.safetensors")
    return directory


class TestTensorDelta(TestCase):
    def test_a_rank_one_delta_reports_one_direction_holding_all_of_the_energy(self):
        pre = torch.eye(ROWS, COLUMNS)
        outer = torch.zeros(ROWS, COLUMNS)
        outer[:, 0] = torch.linspace(1.0, 2.0, ROWS)
        delta = tensor_delta("model.layers.0.mlp.up_proj.weight", pre, pre + outer)
        self.assertAlmostEqual(delta.stable_rank, 1.0, places=4)
        self.assertAlmostEqual(delta.top_k_energy, 1.0, places=4)
        self.assertEqual(delta.rank_for_half, 1)
        self.assertEqual((delta.layer, delta.kind), (0, "up_proj"))

    def test_an_isotropic_delta_reports_the_full_rank_of_its_shape(self):
        pre = torch.zeros(ROWS, COLUMNS)
        # An orthogonal delta has every singular value equal, which is the
        # definition of using every direction the shape allows.
        post = torch.linalg.qr(torch.randn(ROWS, COLUMNS))[0]
        delta = tensor_delta("w", pre, post)
        self.assertEqual(delta.full_rank, COLUMNS)
        self.assertAlmostEqual(delta.stable_rank, float(COLUMNS), places=3)
        self.assertAlmostEqual(delta.concentrated, 1.0, places=3)

    def test_no_movement_leaves_the_rank_questions_unanswered(self):
        pre = torch.eye(ROWS, COLUMNS)
        delta = tensor_delta("w", pre, pre.clone())
        self.assertEqual(delta.delta_relative, 0.0)
        self.assertIsNone(delta.stable_rank)
        self.assertIsNone(delta.top_k_energy)
        self.assertIn("moved 0.0%", str(delta))

    def test_a_vector_has_a_size_but_no_singular_values(self):
        delta = tensor_delta("model.norm.weight", torch.ones(ROWS), torch.ones(ROWS) * 1.5)
        self.assertAlmostEqual(delta.delta_relative, 0.5, places=6)
        self.assertIsNone(delta.stable_rank)
        self.assertIsNone(delta.full_rank)
        self.assertIsNone(delta.concentrated)

    def test_relative_movement_is_scale_free(self):
        small, large = torch.eye(ROWS, COLUMNS), torch.eye(ROWS, COLUMNS) * 100
        step = torch.zeros(ROWS, COLUMNS)
        step[0, 0] = 1.0
        near = tensor_delta("w", small, small + step)
        far = tensor_delta("w", large, large + step * 100)
        self.assertAlmostEqual(near.delta_relative, far.delta_relative, places=5)

    def test_two_shapes_are_not_two_versions_of_one_model(self):
        with self.assertRaises(DiffingError):
            tensor_delta("w", torch.zeros(ROWS, COLUMNS), torch.zeros(COLUMNS, ROWS))

    def test_the_estimate_agrees_with_the_exact_answer_and_says_that_it_is_one(self):
        generator = torch.Generator().manual_seed(0)
        pre = torch.randn(ROWS * 8, COLUMNS * 8, generator=generator)
        post = pre + 0.1 * torch.randn(ROWS * 8, COLUMNS * 8, generator=generator)
        exact = tensor_delta("w", pre, post)
        estimated = tensor_delta("w", pre, post, estimate=True)
        self.assertFalse(exact.estimated)
        self.assertTrue(estimated.estimated)
        self.assertAlmostEqual(exact.stable_rank, estimated.stable_rank, delta=exact.stable_rank * 0.05)
        self.assertIn("~", str(estimated))


class TestClassify(TestCase):
    def test_a_layer_tensor_gives_its_index_and_kind(self):
        self.assertEqual(classify("model.layers.7.self_attn.q_proj.weight"), (7, "q_proj"))
        self.assertEqual(classify("model.layers.21.mlp.down_proj.weight"), (21, "down_proj"))

    def test_everything_off_the_stack_gives_neither(self):
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                     "model.layers.3.input_layernorm.weight"):
            with self.subTest(name=name):
                self.assertEqual(classify(name), (None, None))


class TestFiles(TestCase):
    def test_a_sharded_checkpoint_indexes_the_same_as_a_single_file_one(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            tensors = layer_tensors(0)
            one = checkpoint(root / "one", tensors, shards=1)
            many = checkpoint(root / "many", tensors, shards=3)
            self.assertEqual(sorted(index(one)), sorted(index(many)))
            self.assertEqual(len(sorted(set(index(many).values()))), 3)
            self.assertEqual(shapes(index(one)), shapes(index(many)))

    def test_a_tensor_comes_back_as_float32_whatever_it_was_written_as(self):
        with tempfile.TemporaryDirectory() as workspace:
            root = checkpoint(Path(workspace), {"w": torch.ones(ROWS, COLUMNS, dtype=torch.bfloat16)})
            tensor = read(index(root), "w")
            self.assertEqual(tensor.dtype, torch.float32)

    def test_a_directory_with_no_weights_says_what_it_wanted(self):
        with tempfile.TemporaryDirectory() as workspace:
            with self.assertRaises(DiffingError) as raised:
                index(Path(workspace))
            self.assertIn("safetensors", str(raised.exception))

    def test_a_byte_identical_copy_is_found_and_named(self):
        with tempfile.TemporaryDirectory() as workspace:
            shared = torch.randn(ROWS, COLUMNS)
            root = checkpoint(Path(workspace), {
                "model.embed_tokens.weight": shared,
                "lm_head.weight": shared.clone(),
                "other.weight": torch.randn(ROWS, COLUMNS),
            })
            found = duplicates(index(root), ["lm_head.weight", "other.weight"])
            self.assertEqual(found, {"lm_head.weight": "model.embed_tokens.weight"})


class TestUnexplained(TestCase):
    def test_a_perfect_reconstruction_leaves_nothing(self):
        truth = torch.randn(ROWS, COLUMNS)
        fvu, raw = unexplained(truth, torch.zeros_like(truth))
        self.assertEqual((fvu, raw), (0.0, 0.0))

    def test_reconstructing_the_mean_leaves_exactly_the_variance(self):
        # Predicting each column's mean everywhere is the standard baseline FVU
        # is defined against, so it has to come out at 1.0 by construction.
        truth = torch.randn(ROWS, COLUMNS)
        error = truth - truth.mean(0, keepdim=True)
        fvu, _ = unexplained(truth, error)
        self.assertAlmostEqual(fvu, 1.0, places=5)

    def test_the_two_denominators_differ_when_the_mean_is_not_zero(self):
        truth = torch.ones(ROWS, COLUMNS) * 3 + 0.1 * torch.randn(ROWS, COLUMNS)
        error = 0.1 * torch.randn(ROWS, COLUMNS)
        fvu, raw = unexplained(truth, error)
        # A large offset inflates the squared norm and not the variance, so the
        # uncentred figure is the smaller of the two and calling either "the"
        # FVU without saying which would be a different number each time.
        self.assertGreater(fvu, raw)

    def test_a_flat_signal_has_no_variance_to_explain(self):
        truth = torch.ones(ROWS, COLUMNS)
        fvu, raw = unexplained(truth, torch.zeros_like(truth))
        self.assertTrue(fvu != fvu)  # nan: the question is undefined, not zero
        self.assertEqual(raw, 0.0)


class TestFitShift(TestCase):
    def build(self, first, second):
        return (
            FitReport(checkpoint="a", release="r", layers=[LayerFit(i, v, v, 4.0) for i, v in enumerate(first)]),
            FitReport(checkpoint="b", release="r", layers=[LayerFit(i, v, v, 4.0) for i, v in enumerate(second)]),
        )

    def test_the_shift_is_per_layer_and_signed(self):
        before, after = self.build([0.2, 0.3, 0.1], [0.25, 0.9, 0.05])
        shift = fit_shift(before, after)
        self.assertEqual([row["layer"] for row in shift], [0, 1, 2])
        self.assertAlmostEqual(shift[1]["delta"], 0.6, places=6)
        self.assertLess(shift[2]["delta"], 0)

    def test_a_mean_hides_where_the_dictionary_failed(self):
        # The reason fit_shift returns rows: these two have the same mean shift
        # and only one of them has a layer that stopped being described at all.
        spread, spiked = self.build([0.3] * 4, [0.4] * 4)[1], self.build([0.3] * 4, [0.3, 0.3, 0.3, 0.7])[1]
        base = self.build([0.3] * 4, [0.3] * 4)[0]
        self.assertAlmostEqual(spread.fvu, spiked.fvu, places=6)
        self.assertLess(max(row["delta"] for row in fit_shift(base, spread)), 0.2)
        self.assertGreater(max(row["delta"] for row in fit_shift(base, spiked)), 0.3)

    def test_two_reports_of_different_depth_will_not_be_compared(self):
        before, after = self.build([0.2, 0.3], [0.2, 0.3, 0.4])
        with self.assertRaises(ValueError):
            fit_shift(before, after)


class TestCheckpointDelta(TestCase):
    def build(self, workspace: str):
        generator = torch.Generator().manual_seed(0)
        before = {**layer_tensors(0), **layer_tensors(1), "model.embed_tokens.weight": torch.randn(ROWS, COLUMNS)}
        after = {
            **layer_tensors(0, scale=0.01, generator=generator),
            **layer_tensors(1, scale=0.2, generator=generator),
            "model.embed_tokens.weight": before["model.embed_tokens.weight"].clone(),
        }
        after["lm_head.weight"] = after["model.embed_tokens.weight"].clone()
        root = Path(workspace)
        return (
            checkpoint(root / "pre", before),
            checkpoint(root / "post", after, shards=2),
        )

    def test_the_diff_covers_every_shared_tensor_and_accounts_for_the_rest(self):
        with tempfile.TemporaryDirectory() as workspace:
            pre, post = self.build(workspace)
            report = checkpoint_delta(pre, post)
            self.assertEqual(len(report.deltas), len(index(pre)))
            self.assertEqual(report.layers, [0, 1])
            self.assertIn("lm_head.weight", report.skipped)
            self.assertIn("byte-identical", report.skipped["lm_head.weight"])

    def test_the_layer_that_moved_more_is_the_layer_reported_as_moving_more(self):
        with tempfile.TemporaryDirectory() as workspace:
            pre, post = self.build(workspace)
            report = checkpoint_delta(pre, post)
            profile = report.profile(FEEDFORWARD)
            self.assertLess(profile[0]["delta_relative"], profile[1]["delta_relative"])
            self.assertEqual(profile[0]["n"], len(FEEDFORWARD))

    def test_an_unmoved_tensor_stays_out_of_the_rank_aggregates(self):
        with tempfile.TemporaryDirectory() as workspace:
            pre, post = self.build(workspace)
            report = checkpoint_delta(pre, post)
            embedding = next(delta for delta in report.deltas if delta.name == "model.embed_tokens.weight")
            self.assertEqual(embedding.delta_relative, 0.0)
            self.assertNotIn(embedding, report.matrices)

    def test_every_tensor_is_announced_once(self):
        with tempfile.TemporaryDirectory() as workspace:
            pre, post = self.build(workspace)
            seen = []
            report = checkpoint_delta(pre, post, on_tensor=seen.append)
            self.assertEqual([delta.name for delta in seen], [delta.name for delta in report.deltas])

    def test_the_report_survives_a_round_trip_through_its_artifact(self):
        with tempfile.TemporaryDirectory() as workspace:
            pre, post = self.build(workspace)
            report = checkpoint_delta(pre, post)
            again = DeltaReport.from_dict(report.as_dict())
            self.assertEqual(str(again), str(report))
            self.assertEqual(again.profile(ATTENTION), report.profile(ATTENTION))
            self.assertEqual(again.skipped, report.skipped)

    def test_the_table_has_a_row_per_kind_and_a_row_per_layer(self):
        with tempfile.TemporaryDirectory() as workspace:
            pre, post = self.build(workspace)
            lines = list(table(checkpoint_delta(pre, post)))
            self.assertTrue(lines[0].startswith("kind"))
            self.assertEqual(sum(1 for line in lines if line.startswith(("q_proj", "down_proj"))), 2)
            self.assertIn(f"top{TOP_K}", lines[0])
            self.assertEqual(sum(1 for line in lines if line.strip().startswith(("0", "1"))), 2)
