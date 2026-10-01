"""The text a diff of the SDFT/SFT fine-tunes is measured on, identical for every checkpoint

A model diff compares two checkpoints *on the same input*, so the text cannot be
either model's own continuation: that would compare each model on a different
sequence and call the difference a change in the model. The fixed text here is
the eval prompt, chat-templated the way the eval templated it, followed by the
**base** checkpoint's saved greedy answer -- on-distribution for the starting
point, and the same characters for every fine-tune measured against it.

The answers come from `make eval` in `~/main/Self-Distillation`, which names a
results directory after the checkpoint path with slashes turned into
underscores. Neutral prose is `wikitext.passages`; this module is the other
half, the distribution the fine-tunes were trained on.

A common pipe could be: task_texts(root, "Qwen/Qwen3-0.6B", "tooluse", tokenizer) | capture | fit
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

TASKS = ("tooluse", "science")


class SDFTDataError(ValueError):
    """Saved eval answers that cannot be read, said with the command that would make them"""


@dataclass(frozen=True)
class TaskText:
    """One fixed sequence: a templated prompt and the answer that follows it

    `prompt` is kept apart from `answer` because a measurement may want only the
    answer's positions -- the part a fine-tune was trained to change -- and the
    boundary is a character offset, not something to re-derive by searching.
    """
    task: str
    prompt: str
    answer: str

    @property
    def text(self) -> str:
        return self.prompt + self.answer


def results_name(checkpoint: str, repo: Path) -> str:
    """The directory `make eval` writes for a checkpoint: a hub name or a path inside the repo"""
    path = Path(checkpoint)
    if path.is_absolute():
        try:
            checkpoint = str(path.relative_to(repo))
        except ValueError as error:
            raise SDFTDataError(
                f"{checkpoint} is outside {repo}, so `make eval` never named a directory after it") from error
    return checkpoint.rstrip("/").replace("/", "_")


def task_texts(repo: Path, checkpoint: str, task: str, tokenizer, budget: int,
               limit: Optional[int] = None) -> List[TaskText]:
    """A checkpoint's saved eval answers on one task, each behind its templated prompt

    The science eval saves its prompt as messages and templates it at generation
    time; the tool-use eval saves the templated string. Both come back here as the
    string the model actually read.
    """
    if task not in TASKS:
        raise SDFTDataError(f"task is one of {TASKS}, got '{task}'")
    path = Path(repo) / "results" / results_name(checkpoint, Path(repo)) / f"{task}-{budget}" / "eval_responses.json"
    if not path.exists():
        raise SDFTDataError(f"{path} does not exist; run `make eval DATASET={task} CKPT={checkpoint}` in {repo}")
    texts = []
    for item in json.loads(path.read_text())[:limit]:
        prompt = item["prompt"]
        if not isinstance(prompt, str):
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        if item["response"]:
            texts.append(TaskText(task=task, prompt=prompt, answer=item["response"]))
    return texts


def training_texts(repo: Path, task: str, tokenizer) -> List[str]:
    """Every training demonstration of one task as a templated conversation, the way both methods saw it

    The prompt is the one the SDFT student and the SFT model read, and the
    assistant turn is the demonstration: the SFT target and the text inside the
    SDFT teacher's prompt. For a crosscoder this is the in-distribution half of
    the training stream; nothing here is in either task's eval split.
    """
    from datasets import load_from_disk

    if task not in TASKS:
        raise SDFTDataError(f"task is one of {TASKS}, got '{task}'")
    path = Path(repo) / "data" / f"{task}_data" / "train_data"
    if not path.exists():
        raise SDFTDataError(f"{path} does not exist; the SDFT repo ships both tasks' train splits under data/")
    texts = []
    for row in load_from_disk(str(path)):
        if task == "tooluse":
            messages = [{"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": "\n".join(row["golden_response"])}]
        else:
            messages = [*row["messages"], {"role": "assistant", "content": row["output_text"]}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def eval_items(repo: Path, task: str) -> List[dict]:
    """One task's eval split as {messages, answer}: what the SDFT repo's eval script reads and scores against"""
    from datasets import load_from_disk

    if task not in TASKS:
        raise SDFTDataError(f"task is one of {TASKS}, got '{task}'")
    rows = load_from_disk(str(Path(repo) / "data" / f"{task}_data" / "eval_data")).to_list()
    if task == "tooluse":
        return [{"messages": [{"role": "user", "content": row["prompt"]}], "answer": row["golden_answer"]}
                for row in rows]
    return [{"messages": row["prompt"], "answer": row["answer"]} for row in rows]


def score(task: str, response: str, answer) -> int:
    """1 if a response is correct by the SDFT repo's own rule, else 0

    A port of `eval_tooluse.evaluate_correctness` and `eval_science.evaluate_correctness`
    (those modules import vLLM at the top, so they cannot be imported here).
    Tool use: the actions after `</think>` must match the golden actions as a
    multiset and the merged action inputs must match exactly. Science: the text
    inside the last `<answer>` must equal the letter. The port is checked against
    the repo's saved `correct` flags on every saved answer.
    """
    import re
    from collections import Counter
    from contextlib import suppress

    if task == "science":
        extracted = response.split("<answer>")[-1].split("</answer>")[0].strip()
        return int(extracted == answer)
    tail = response.split("</think>")[-1]
    predicted_actions = re.findall(r"Action:\s*(\w+)", tail)
    predicted_inputs = {}
    for block in re.findall(r"Action Input:\s*({.*?})", tail, re.DOTALL):
        try:
            predicted_inputs.update(json.loads(block))
        except json.JSONDecodeError:
            continue
    golden_actions = [item["Action"] for item in answer]
    golden_inputs = {}
    for item in answer:
        with suppress(json.JSONDecodeError, TypeError):
            golden_inputs.update(json.loads(item["Action_Input"]))
    return int(Counter(predicted_actions) == Counter(golden_actions) and predicted_inputs == golden_inputs)
