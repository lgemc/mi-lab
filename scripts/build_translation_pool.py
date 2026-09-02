"""Build the word-level es->en pool from a real lexicon instead of a typed list.

`WORD_PAIRS` in data/translation.py is forty pairs someone wrote down. On
Qwen3-1.7B twenty-five of them survive the tokenizer, and a 75/25 split turns
that into eighteen training prompts -- against which the whole-model sheaf put
28M open gates and memorized, held-out 0.125 against train 0.917. No number of
steps fixes a pool of eighteen. This is the pool.

The source is MUSE (Conneau et al., 1710.04087), the bilingual dictionaries
released with the word-translation work and the standard lexicon for exactly
this: one Spanish word, its English translation, ordered by source frequency.
112k lines, of which the useful fraction is small and the filtering is the
whole job.

Four filters, and each one is a way the task breaks without it:

  single token    both sides, leading space included, or the answer is not one
                  logit and the frame is not one position
  unique English  build_translation draws the distractor from another pair, so
                  two Spanish words sharing an English side eventually produce
                  an example whose distractor IS the answer
  no identity     `del -> del`, `final -> final`: nothing is being translated,
                  and the model can answer by copying
  no function     `que -> that`, `hasta -> even`. "The Spanish word que means"
  words           is not a translation question, and MUSE's top-ranked gloss
                  for a function word is frequently a minor sense

  no near-       `numero -> number` and `numeros -> numbers` are one lexical
  duplicates     item. Left in, 104 of 506 pairs shared a stem, and one of each
                 pair would land in train and the other in the holdout -- which
                 is precisely the leak the prompt-wise split was added to close,
                 walked back in through the data. One pair per lemma, both sides.

Then the model itself filters: a pair is kept only if its answer token is the
argmax over every other English token in the pool at the frame's last position.
MUSE's first gloss is not always the right one (`parte -> party` is simply
wrong) and a pair the model does not know is noise in a task about what the
model knows.

That check has to be run with `GENERIC` already removed, and this is not a
detail. Scored against a pool still containing `something`, that one word won
267 of 402 rejections -- 66% -- because "The Spanish word X means" has a
contentless continuation that outscores any actual translation. It was vetoing
correct pairs (`ingles -> english`, `espanol -> spanish`) for a reason with
nothing to do with translation. With the generic glosses dropped the rejection
tail is flat, no word above 31 of 296, which is what a filter measuring the
thing it claims to measure looks like.

Selecting on model behaviour is not circular here: it selects on the *unmasked*
model, before any gate exists, and the question the sheaf asks is which weights
carry a behaviour the model has. It is the same move IOI makes by construction.
What it does mean is that the pool is model-defined rather than a gold lexicon:
a few glosses survive that a lexicographer would argue with (`ciudad -> town`,
`equipo -> equipment`), kept because the model links those tokens strongly and
the task is about the mapping the model has, not the one a dictionary prefers.

Downloaded and derived data stays under data/external/ and out of git, so this
script is the provenance. MUSE's dictionaries are CC BY-NC 4.0.

A common pipe could be: muse | lexical filters | model check | pool | task

Run: uv run python -m scripts.build_translation_pool qwen3-1.7b
"""

import sys
import urllib.request
from pathlib import Path

import torch

from src.data.tasks import single_tokens
from src.data.translation import WORD_FRAME, pool_path
from src.model.adapter import load_adapter
from src.telemetry.observe import Progress, banner, log

MUSE = "https://dl.fbaipublicfiles.com/arrival/dictionaries/es-en.txt"
EXTERNAL = Path("data/external/translation")
CACHE = EXTERNAL / "muse-es-en.txt"

# Function words on either side. The frame asks what a word *means*, which is a
# question about content words; a gloss for `que` or `hasta` is a grammatical
# fact and MUSE's top-ranked one is often a minor sense of it.
SPANISH_FUNCTION = {
    "que", "los", "las", "con", "una", "uno", "unos", "unas", "para", "por", "como", "fue",
    "más", "mas", "sus", "también", "entre", "este", "esta", "estos", "estas", "pero", "son",
    "sobre", "desde", "hasta", "sin", "donde", "cuando", "ser", "estar", "haber", "tener",
    "del", "les", "nos", "eso", "esa", "ese", "aquí", "allí", "muy", "todo", "toda", "todos",
    "todas", "otro", "otra", "otros", "otras", "cual", "cuales", "quien", "quienes", "porque",
    "aunque", "mientras", "según", "hacia", "bajo", "ante", "tras", "durante", "sino", "pues",
    "así", "ya", "aún", "aun", "solo", "sólo", "aquel", "aquella", "cada", "algún", "alguna",
    "ningún", "ninguna", "aquellos", "aquellas", "aquello", "había", "debido", "junto",
    "dentro", "poco", "veces",
}
ENGLISH_FUNCTION = {
    "that", "the", "with", "one", "for", "how", "was", "more", "their", "also", "among",
    "this", "but", "are", "about", "since", "even", "without", "where", "when", "being",
    "and", "not", "have", "has", "had", "were", "been", "from", "into", "than", "then",
    "there", "these", "those", "which", "who", "whom", "whose", "what", "some", "any",
    "each", "every", "such", "only", "just", "very", "much", "many", "both", "either",
    "neither", "other", "another", "same", "own", "all", "its", "his", "her", "our",
    "your", "they", "them", "you", "she", "him", "would", "could", "should", "will",
    "can", "may", "might", "must", "shall", "did", "does", "done", "made", "make",
    "due", "along", "next", "inside", "late", "well", "literally", "times",
}

# Contentless glosses. Not translations anyone is testing for, and at this frame
# they are attractors: with `something` left in the pool it won 66% of all
# rejections and vetoed correct pairs on its own.
GENERIC = {
    "something", "anything", "nothing", "thing", "things", "someone", "somebody",
    "stuff", "kind", "sort", "way", "ways", "form", "forms",
}

MIN_LETTERS = 3
BATCH = 64

def lemma(word: str) -> str:
    """Crude and targeted: strip the plural endings that produced the duplicates

    Not a stemmer and not trying to be. It exists to stop `numero`/`numeros`
    and `sistema`/`sistemas` from being two pool entries, which is the only
    near-duplicate shape MUSE's frequency ordering actually produced here.
    """
    for ending in ("es", "s"):
        if len(word) > 4 and word.endswith(ending):
            return word[: -len(ending)]
    return word

def fetch() -> Path:
    """MUSE es-en, cached, because 112k lines is not worth downloading twice"""
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        log(f"downloading {MUSE}")
        with urllib.request.urlopen(MUSE, timeout=120) as response:
            CACHE.write_bytes(response.read())
    return CACHE

def candidates(adapter) -> list:
    """MUSE, reduced to pairs the frame and the tokenizer can both carry"""
    seen_source, seen_target, kept = set(), set(), []
    seen_source_lemma, seen_target_lemma = set(), set()
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        spanish, english = parts[0].strip().lower(), parts[1].strip().lower()
        if spanish in seen_source:
            continue  # MUSE lists several glosses per source; the first is the best ranked
        seen_source.add(spanish)
        if not (spanish.isalpha() and english.isalpha()):
            continue
        if spanish == english:
            continue
        if len(spanish) < MIN_LETTERS or len(english) < MIN_LETTERS:
            continue
        if spanish in SPANISH_FUNCTION or english in ENGLISH_FUNCTION or english in GENERIC:
            continue
        if english in seen_target:
            continue  # a repeated English side can collide with the distractor
        if lemma(spanish) in seen_source_lemma or lemma(english) in seen_target_lemma:
            continue  # one lexical item per pool, or it straddles the train/holdout split
        if len(single_tokens(adapter, (f" {spanish}", f" {english}"))) != 2:
            continue
        seen_target.add(english)
        seen_source_lemma.add(lemma(spanish))
        seen_target_lemma.add(lemma(english))
        kept.append((spanish, english))
    return kept

def verified(adapter, pairs: list) -> list:
    """Keep the pairs whose answer beats every other English word in the pool

    One forward pass per prompt, batched, comparing only the pool's own answer
    tokens. Beating the whole vocabulary would be a different and much harsher
    test -- the argmax at this frame is often a quote character, because the
    natural continuation is `means "table"` -- and it is not the test the task
    applies. The task is a two-way choice against another pool word, so this
    checks exactly that, against all of them at once.

    Depends on `GENERIC` having already been dropped upstream; see the module
    docstring for what one contentless gloss did to this check.
    """
    answers = [adapter.single_token(f" {english}") for _, english in pairs]
    ids = torch.tensor(answers)
    prompts = [WORD_FRAME.format(source=f" {spanish}", answer="") for spanish, _ in pairs]
    kept, progress = [], Progress(len(prompts), "verify", every=max(1, len(prompts) // 10))
    for start in range(0, len(prompts), BATCH):
        chunk = prompts[start : start + BATCH]
        logits = adapter.logits(chunk)
        over_pool = logits[:, ids]
        winners = over_pool.argmax(dim=-1)
        for offset, winner in enumerate(winners.tolist()):
            row = start + offset
            if winner == row:
                kept.append(pairs[row])
            progress.tick()
    progress.finish()
    return kept

def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
    fetch()
    adapter = load_adapter(config)
    banner("translation pool", {
        "config": config,
        "source": MUSE,
        "cache": CACHE,
        "output": pool_path(config),
    })
    pairs = candidates(adapter)
    log(f"{len(pairs)} pairs survive the lexical and tokenizer filters")
    kept = verified(adapter, pairs)
    log(f"{len(kept)} of those the model answers correctly against the whole pool")

    path = pool_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# es->en word pool for {config}, from MUSE ({MUSE})",
             f"# {len(kept)} pairs: single-token both sides, unique English side, "
             f"model-verified against the pool",
             "# rebuild: uv run python -m scripts.build_translation_pool " + config]
    lines.extend(f"{spanish}\t{english}" for spanish, english in kept)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    split = int(len(kept) * 0.75)
    log(f"-> {path}")
    log(f"a 75/25 split by prompt gives {split} train / {len(kept) - split} held-out, "
        f"against 18 / 7 from the built-in list")

if __name__ == "__main__":
    main()
