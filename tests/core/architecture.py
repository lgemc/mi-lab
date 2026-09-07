import re
import unittest
from pathlib import Path

"""
The second axis, as a test: which words the kernel is allowed to know.

`tests/core/config.py::TestNoHardcodedModelFacts` greps src/ for widths and
fails on one in a docstring, and it works precisely because it reads raw text:
a module whose prose explains itself in terms of 768 is a module about one
checkpoint. This is the same instrument pointed at modality vocabulary.

A concept belongs in the kernel if it survives the substitution *model -> a
differentiable function with addressable internal sites, scored by a scalar
readout*. Nothing that survives that substitution needs to say "token",
"prompt", "vocab", "logit", "tokenizer", "word" or "sentence". Everything that
does is either a domain module in the wrong place or a domain assumption
wearing the clothes of an invariant.

There are false positives and they are kept on purpose: `methods/common/
components.py` says "the component vocabulary" and `share/schema/artifact.py`
says "a sentence with a meaning", and neither is about language. Narrowing the
word list to dodge them would also dodge the leaks. The allow-list absorbs them
instead, and the property that matters is not that the list is short -- it is
that it is *exact*, so it can only be changed deliberately.

LEAKS is therefore compared for equality, not for containment. A file that
starts naming a modality fails, and so does a file that stopped: the table only
shrinks when somebody refreshes it, which is what makes the shrink visible.

Refresh it with, and only ever deliberately:

    uv run python -m tests.core.architecture --refresh
"""

SOURCE = Path(__file__).resolve().parents[2] / "src"

#: Directories the rule does not apply to, and the reason for each.
#:
#: `domains/` is the other axis -- a domain is where knowing what a token is
#: belongs. `cli/` prints for a human who typed a prompt, so it is allowed to
#: name one; it formats what the kernel returns and computes nothing.
EXEMPT = ("domains", "cli")

#: The modality vocabulary, by stem, with the inflections that actually appear.
#: Matching is on whole alphabetic runs rather than on substrings, so
#: `token_labels` counts and `keyword` does not.
FORMS = {
    "token": ("token", "tokens"),
    "prompt": ("prompt", "prompts", "prompted", "prompting"),
    "vocab": ("vocab", "vocabulary", "vocabularies"),
    "logit": ("logit", "logits"),
    "tokenizer": (
        "tokenizer", "tokenizers", "tokenize", "tokenized", "tokenizes", "tokenizing", "tokenization",
    ),
    "word": ("word", "words"),
    "sentence": ("sentence", "sentences"),
}

_STEM = {form: stem for stem, forms in FORMS.items() for form in forms}


def modality_words(text: str) -> set:
    """Every modality stem this text names, read as raw text rather than as code"""
    return {_STEM[piece.lower()] for piece in re.split(r"[^A-Za-z]+", text) if piece.lower() in _STEM}


def scan_kernel(root: Path = SOURCE) -> dict:
    """Every kernel module that names a modality, and which stems it names"""
    found = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if relative.parts[0] in EXEMPT:
            continue
        words = modality_words(path.read_text())
        if words:
            found[relative.as_posix()] = sorted(words)
    return found


# --- the frozen table: every leak in the kernel today ---------------------
#
# Written down so it cannot grow, and so the phases that shrink it have
# something to shrink. Regenerate with the command in the module docstring;
# the two markers are what the refresh rewrites between.
# BEGIN LEAKS
LEAKS = {
    "core/config.py": ["prompt", "token"],
    "core/metrics.py": ["logit", "prompt", "sentence", "token", "vocab"],
    "core/readout.py": ["logit", "token", "tokenizer"],
    "data/dataset.py": ["prompt", "sentence", "token", "word"],
    "data/prompts.py": ["prompt", "sentence", "tokenizer", "word"],
    "data/tasks.py": ["logit", "prompt", "sentence", "token", "tokenizer", "vocab", "word"],
    "data/torchdata.py": ["prompt", "sentence", "token", "tokenizer", "vocab"],
    "experiment/pipeline.py": ["sentence"],
    "experiment/runner.py": ["prompt", "token"],
    "experiment/sheaf.py": ["prompt"],
    "experiment/spec.py": ["prompt"],
    "ie/app.py": ["prompt", "token"],
    "ie/command.py": ["prompt", "sentence", "token", "word"],
    "ie/main.py": ["prompt", "token"],
    "ie/registry.py": ["token"],
    "ie/session.py": ["prompt", "token"],
    "ie/view.py": ["token", "vocab"],
    "ie/views/activations.py": ["prompt", "token"],
    "ie/views/grid.py": ["token"],
    "ie/views/models.py": ["token"],
    "ie/views/tokens.py": ["prompt", "sentence", "token", "tokenizer"],
    "methods/circuits/ablation.py": ["logit", "prompt", "sentence"],
    "methods/circuits/attribution.py": ["logit", "prompt"],
    "methods/circuits/comparison.py": ["logit", "prompt", "sentence"],
    "methods/circuits/faithfulness.py": ["logit", "prompt", "token"],
    "methods/circuits/patching.py": ["sentence", "token"],
    "methods/circuits/techniques.py": ["logit", "sentence"],
    "methods/circuits/wiring.py": ["vocab"],
    "methods/common/components.py": ["vocab"],
    "methods/common/errors.py": ["vocab", "word"],
    "methods/common/intervention.py": ["token"],
    "methods/common/span.py": ["logit", "prompt", "sentence", "token"],
    "methods/knockout/ablate.py": ["logit", "prompt", "sentence", "token"],
    "methods/knockout/cost.py": ["prompt", "token"],
    "methods/knockout/neurons.py": ["prompt", "sentence", "token"],
    "methods/probing/probe.py": ["logit"],
    "methods/probing/steering.py": ["prompt", "sentence", "token", "word"],
    "methods/sheaves/faith.py": ["logit", "sentence", "token", "vocab", "word"],
    "methods/sheaves/forward.py": ["logit", "prompt", "token", "tokenizer", "vocab", "word"],
    "methods/sheaves/gate.py": ["logit"],
    "methods/sheaves/gateable.py": ["logit", "sentence", "token"],
    "methods/sheaves/mask.py": ["logit", "prompt", "token", "tokenizer", "vocab", "word"],
    "methods/sheaves/training.py": ["logit", "prompt", "token", "vocab", "word"],
    "methods/sheaves/units.py": ["logit", "vocab"],
    "model/adapter.py": ["logit", "prompt", "token", "tokenizer", "vocab", "word"],
    "model/passes.py": ["prompt", "token", "tokenizer"],
    "serve/app.py": ["prompt", "token", "word"],
    "serve/backbones.py": ["logit", "token"],
    "serve/circuits.py": ["prompt", "sentence", "token", "word"],
    "serve/examples.py": ["prompt"],
    "share/converters/activations.py": ["token", "vocab"],
    "share/converters/circuit.py": ["logit", "prompt", "token", "vocab"],
    "share/converters/comparison.py": ["logit", "prompt", "token", "vocab"],
    "share/converters/probe.py": ["logit"],
    "share/converters/steering.py": ["vocab", "word"],
    "share/definitions.py": ["logit", "prompt", "sentence"],
    "share/schema/artifact.py": ["logit", "sentence", "token", "vocab"],
    "share/schema/control.py": ["logit"],
    "share/schema/metric.py": ["logit", "sentence"],
    "share/schema/node.py": ["logit", "token", "vocab"],
    "share/schema/payload.py": ["logit", "prompt", "token", "word"],
    "share/schema/site.py": ["vocab"],
    "share/schema/span.py": ["logit"],
    "share/schema/vocabulary.py": ["token", "vocab"],
    "share/storage.py": ["sentence", "vocab"],
    "telemetry/tracking.py": ["token"],
    "viz/activations.py": ["prompt", "token"],
    "viz/circuits.py": ["logit", "sentence", "token"],
    "viz/comparison.py": ["logit"],
    "viz/dataset.py": ["sentence", "token", "tokenizer", "vocab", "word"],
    "viz/steering.py": ["sentence", "word"],
}
# END LEAKS


class TestKernelNamesNoModality(unittest.TestCase):
    def test_the_leaks_are_exactly_the_ones_written_down(self):
        """A kernel module that starts naming a modality is a kernel module about one"""
        found = scan_kernel()
        added = {path: words for path, words in found.items() if LEAKS.get(path) != words}
        gone = {path: words for path, words in LEAKS.items() if path not in found}
        self.assertEqual(
            LEAKS, found,
            "the modality allow-list no longer matches src/.\n"
            f"  now naming a modality, or naming a new one: {sorted(added)}\n"
            f"  no longer naming one (refresh the table): {sorted(gone)}\n"
            "  refresh with: uv run python -m tests.core.architecture --refresh",
        )

    def test_the_exempt_directories_exist(self):
        """An allow-list keyed on a directory that is gone allows everything, quietly"""
        for name in EXEMPT:
            with self.subTest(directory=name):
                self.assertTrue((SOURCE / name).is_dir(), f"src/{name}/ is in EXEMPT and does not exist")

    def test_a_word_list_that_matches_nothing_would_be_a_check_that_cannot_fail(self):
        """The instrument itself, on a line that is unambiguously about language"""
        self.assertEqual({"logit", "token"}, modality_words("d(logit(io)) / d(token)"))
        self.assertEqual(set(), modality_words("a keyword argument, password protected"))


def _refresh() -> None:
    """Rewrite LEAKS in this file from what src/ says today"""
    path = Path(__file__)
    found = scan_kernel()
    body = "LEAKS = {\n" + "".join(
        f'    "{name}": {words!r},\n'.replace("'", '"') for name, words in found.items()
    ) + "}\n"
    text = re.sub(
        r"(?<=# BEGIN LEAKS\n).*?(?=# END LEAKS\n)", body, path.read_text(), flags=re.S,
    )
    path.write_text(text)
    print(f"wrote {len(found)} entries to {path}")


if __name__ == "__main__":
    import sys

    if "--refresh" in sys.argv:
        _refresh()
    else:
        unittest.main()
