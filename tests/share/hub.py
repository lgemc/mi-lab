import json
import tempfile
from pathlib import Path
from unittest import TestCase

import torch
from safetensors.torch import load_file, save_file

from src.share.hub import HubError, checkpoint_name, crosscoder_plan, plan, single_file, stage_name

"""
The Hub layout is tested on a fake `runs/` tree laid out the way the
Self-Distillation Makefile writes one, with a few-byte tensor standing in for
each model. What can go wrong is all naming and refusal: a stage published
under its last dataset instead of its sequence, SDFT and SFT landing on one
path, a half-finished stage shipped from its checkpoint, or one method pushed
without the other. `push` itself is the Hub's client and is not exercised.

Crosscoders are named through the configs they record, so those tests hand
`crosscoder_plan` a resolver over the fake tree instead of `configs/`.
"""


def _stage(root: Path, name: str, weights: bool = True) -> Path:
    stage = root / name
    stage.mkdir(parents=True)
    if weights:
        save_file({"w": torch.ones(2)}, str(stage / "model.safetensors"))
    return stage


def _runs(directory: Path, model: str = "Qwen3-0.6B") -> Path:
    runs = directory / "runs"
    for key in ("seq", "sft-seq"):
        _stage(runs / f"{key}-{model}", "1-tooluse")
        _stage(runs / f"{key}-{model}", "2-science")
    return runs


class TestPlan(TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_a_stage_is_named_by_the_sequence_that_produced_it(self):
        self.assertEqual(stage_name(["tooluse"]), "tool-use")
        self.assertEqual(stage_name(["tooluse", "science"]), "tool-use-then-science")

    def test_every_stage_of_both_methods_gets_its_own_path(self):
        paths = sorted(upload.path_in_repo for upload in plan(_runs(self.root)))
        self.assertEqual(
            paths,
            [
                "self-distill-vs-sft/qwen3-0.6b/sdft/tool-use-then-science.safetensors",
                "self-distill-vs-sft/qwen3-0.6b/sdft/tool-use.safetensors",
                "self-distill-vs-sft/qwen3-0.6b/sft/tool-use-then-science.safetensors",
                "self-distill-vs-sft/qwen3-0.6b/sft/tool-use.safetensors",
            ],
        )

    def test_the_final_model_is_published_not_a_checkpoint(self):
        runs = _runs(self.root)
        _stage(runs / "seq-Qwen3-0.6B" / "2-science", "checkpoint-100")
        sources = {upload.path_in_repo: upload.sources for upload in plan(runs)}
        source = sources["self-distill-vs-sft/qwen3-0.6b/sdft/tool-use-then-science.safetensors"]
        self.assertEqual(source, (str(runs / "seq-Qwen3-0.6B" / "2-science" / "model.safetensors"),))

    def test_an_unfinished_stage_is_refused_even_with_a_checkpoint(self):
        runs = self.root / "runs"
        _stage(runs / "seq-Qwen3-0.6B", "1-tooluse")
        unfinished = _stage(runs / "seq-Qwen3-0.6B", "2-science", weights=False)
        _stage(unfinished, "checkpoint-100")
        _stage(runs / "sft-seq-Qwen3-0.6B", "1-tooluse")
        with self.assertRaisesRegex(HubError, "did not finish.*checkpoint-100"):
            plan(runs)

    def test_one_method_alone_is_refused(self):
        runs = self.root / "runs"
        _stage(runs / "seq-Qwen3-0.6B", "1-tooluse")
        with self.assertRaisesRegex(HubError, "no sft run"):
            plan(runs)

    def test_a_gap_in_the_stages_is_refused(self):
        runs = _runs(self.root)
        _stage(runs / "sft-seq-Qwen3-0.6B", "4-medical")
        with self.assertRaisesRegex(HubError, "gap"):
            plan(runs)

    def test_unrelated_runs_are_ignored(self):
        runs = _runs(self.root)
        _stage(runs / "smoke-tooluse-Qwen3-0.6B", "1-tooluse")
        _stage(runs / "tooluse-Qwen3-0.6B", "1-tooluse")
        self.assertEqual(len(plan(runs)), 4)

    def test_shards_are_merged_into_one_file(self):
        runs = self.root / "runs"
        for key in ("seq", "sft-seq"):
            _stage(runs / f"{key}-Qwen3-0.6B", "1-tooluse")
        sharded = runs / "seq-Qwen3-0.6B" / "1-tooluse"
        (sharded / "model.safetensors").unlink()
        save_file({"a": torch.zeros(3)}, str(sharded / "model-00001-of-00002.safetensors"))
        save_file({"b": torch.ones(4)}, str(sharded / "model-00002-of-00002.safetensors"))
        (sharded / "model.safetensors.index.json").write_text(
            json.dumps(
                {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}
            )
        )
        upload = next(u for u in plan(runs) if u.method == "sdft")
        self.assertEqual(len(upload.sources), 2)
        merged = load_file(str(single_file(upload, self.root)))
        self.assertEqual(sorted(merged), ["a", "b"])
        self.assertTrue(torch.equal(merged["b"], torch.ones(4)))


def _crosscoder(results: Path, directory: str, pre: str, post: str, json_too: bool = True) -> Path:
    folder = results / directory
    folder.mkdir(parents=True)
    save_file({"W_dec": torch.zeros(2, 2, 3)}, str(folder / "phase2-crosscoder.safetensors"))
    if json_too:
        (folder / "phase2-crosscoder.json").write_text(json.dumps({"options": {"pre": pre, "post": post}, "layer": 22}))
    return folder


class TestCrosscoderPlan(TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.runs = _runs(self.root)
        self.results = self.root / "results"
        self.configs = {
            "base": "Qwen/Qwen3-0.6B",
            "sdft-tooluse": str(self.runs / "seq-Qwen3-0.6B" / "1-tooluse"),
            "sdft-science": str(self.runs / "seq-Qwen3-0.6B" / "2-science"),
            "sft-science": str(self.runs / "sft-seq-Qwen3-0.6B" / "2-science"),
        }

    def tearDown(self):
        self.directory.cleanup()

    def _plan(self):
        return crosscoder_plan(self.results, resolve=self.configs.__getitem__)

    def test_a_checkpoint_is_named_the_way_its_model_upload_is(self):
        self.assertEqual(checkpoint_name("Qwen/Qwen3-0.6B"), ("qwen3-0.6b", None, "base"))
        self.assertEqual(checkpoint_name(self.configs["sdft-science"]), ("qwen3-0.6b", "sdft", "tool-use-then-science"))

    def test_each_crosscoder_gets_a_folder_named_by_its_pair(self):
        _crosscoder(self.results, "diff-a", "base", "sdft-science")
        _crosscoder(self.results, "diff-b", "sdft-tooluse", "sdft-science")
        _crosscoder(self.results, "diff-c", "base", "sft-science")
        self.assertEqual(
            sorted(upload.path_in_repo for upload in self._plan()),
            [
                "crosscoders/qwen3-0.6b/sdft/base--tool-use-then-science/crosscoder.json",
                "crosscoders/qwen3-0.6b/sdft/base--tool-use-then-science/crosscoder.safetensors",
                "crosscoders/qwen3-0.6b/sdft/tool-use--tool-use-then-science/crosscoder.json",
                "crosscoders/qwen3-0.6b/sdft/tool-use--tool-use-then-science/crosscoder.safetensors",
                "crosscoders/qwen3-0.6b/sft/base--tool-use-then-science/crosscoder.json",
                "crosscoders/qwen3-0.6b/sft/base--tool-use-then-science/crosscoder.safetensors",
            ],
        )

    def test_weights_without_their_report_are_refused(self):
        _crosscoder(self.results, "diff-a", "base", "sdft-science", json_too=False)
        with self.assertRaisesRegex(HubError, "k are unknown"):
            self._plan()

    def test_a_pair_across_methods_is_refused(self):
        _crosscoder(self.results, "diff-a", "sdft-tooluse", "sft-science")
        with self.assertRaisesRegex(HubError, "across two methods"):
            self._plan()

    def test_two_crosscoders_for_one_pair_are_refused(self):
        _crosscoder(self.results, "diff-a", "base", "sdft-science")
        _crosscoder(self.results, "rerun/diff-a", "base", "sdft-science")
        with self.assertRaisesRegex(HubError, "overwrite each other"):
            self._plan()

    def test_an_empty_results_root_is_refused(self):
        self.results.mkdir()
        with self.assertRaisesRegex(HubError, "holds no"):
            self._plan()
