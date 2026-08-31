"""Convert the downloaded FLORES / WMT bitexts into the repo's prompt format.

This script is the provenance of the .prompts files under
data/external/translation/ (which are gitignored, like every converted
download): the FLORES-200 es-en dev parquet (mirror
ignacioct/flores200_es_en_dev_test of the gated facebook/flores repo) and
newstest2013 es-en, the Spanish dev set shipped with WMT14, fetched through
sacrebleu. Rerunning it rebuilds both files byte for byte.

Run: uv run python -m scripts.build_translation_data
"""

import csv
from pathlib import Path

import pandas as pd

from src.data.prompts import save_prompts
from src.data.translation import default_pairs_path, pairs_to_prompts

EXTERNAL = Path("data/external/translation")
WMT_SUBSET = 500

def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    parquet = EXTERNAL / "flores200-es-en-dev.parquet"
    if not parquet.exists():
        from huggingface_hub import hf_hub_download

        fetched = hf_hub_download(
            "ignacioct/flores200_es_en_dev_test", "data/dev-00000-of-00001.parquet", repo_type="dataset"
        )
        pd.read_parquet(fetched).to_parquet(parquet)
    flores = pd.read_parquet(parquet)
    pairs = list(zip(flores["es_sentence"], flores["en_sentence"], strict=True))
    path = default_pairs_path("flores-es-en-dev")
    save_prompts(pairs_to_prompts(pairs, "flores-es-en-dev"), str(path))
    print(f"{path}: {len(pairs)} pairs")

    with (EXTERNAL / "newstest2013-es-en.tsv").open(newline="") as handle:
        rows = [(row[0], row[1]) for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)]
    subset = rows[:WMT_SUBSET]
    path = default_pairs_path(f"wmt-newstest2013-es-en-{WMT_SUBSET}")
    save_prompts(pairs_to_prompts(subset, f"wmt-newstest2013-es-en-{WMT_SUBSET}"), str(path))
    print(f"{path}: {len(subset)} pairs (of {len(rows)})")

if __name__ == "__main__":
    main()
