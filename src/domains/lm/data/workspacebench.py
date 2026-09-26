import json
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from torch.utils.data import DataLoader, Dataset

from ....data.dataset import DatasetError

"""
WorkspaceBench as a torch Dataset: one item bank of
github.com/camilablank/workspace-bench (Blank, Bhatia, Ong and Nanda, 2026),
fetched over HTTPS at construction and held in memory. The banks ask whether
an activation-reading lens surfaces what a model computed but never wrote --
an item names a prompt, the `intermediates` the model must have held, and the
readout position -- and each bank carries the `gate` a judge scores a readout
against.

Nothing is downloaded to disk: the whole benchmark is 3057 items of JSON, the
largest bank is under a megabyte, and a file under data/external is a file
that can go stale against the commit it was taken from. COMMIT pins which
version is fetched, so two runs a month apart read the same items.

An item is yielded as the dict it is in the bank. The 26 banks are 26 schemas
-- association carries prompt/intermediates/readout, moral_rationale
stimulus/question, agentic_misalignment system/text -- so a common row shape
would be one this benchmark does not have, and `collate_items` keeps them
dicts rather than letting the default collate stack ragged fields.

A common pipe could be: WorkspaceBench | item_loader | adapter.capture
"""

COMMIT = "92d763e722377f5bfae045e829a308ab5919a820"
RAW = "https://raw.githubusercontent.com/camilablank/workspace-bench/{ref}/evals/{family}/items.json"

# The banks that hold items. `baselines` (scores) and `jlens_concept_pr`
# (per-layer readouts) live under evals/ too and are not item banks.
FAMILIES = (
    "agentic_misalignment", "arithmetic_intermediates", "association", "basic_readout",
    "basic_readout_mt", "brew_intermediates", "buggy_code", "chain_intermediates",
    "conjunctive_association", "directed_modulation", "hallucination", "jailbreak_recognition",
    "moral_rationale", "multi_concept_directed_modulation", "multihop", "multihop_mt",
    "multilingual", "multilingual_mt", "multilingual_multihop", "multilingual_typo", "poetry",
    "relational_multihop", "role_bound_association", "typo", "typo_mt", "user_modeling",
)

# The banks a single-token readout can be asked at all: their items name a
# prompt, the intermediates to find in it, and a read position that is one
# token. The other sixteen either want prose back (moral_rationale,
# agentic_misalignment) or carry no intermediates to score against, and
# running a lens on them produces a zero that says nothing about the lens.
SINGLE_TOKEN_BANKS = (
    "association", "basic_readout", "basic_readout_mt", "multihop", "multihop_mt", "multilingual",
    "multilingual_mt", "multilingual_multihop", "multilingual_typo", "typo", "typo_mt",
)
SINGLE_TOKEN_KINDS = ("final_prompt_token", "last_word_token")

def readable(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The items of a bank that a one-token readout can be scored on

    An item qualifies by carrying all three of the things the scoring needs --
    a prompt to run, intermediates to look for, and a read position naming a
    single token -- rather than by belonging to a bank named above, so a bank
    that mixes read positions loses only the items that do not fit.
    """
    return [item for item in items
            if isinstance(item.get("prompt"), str) and item.get("intermediates")
            and (item.get("readout") or {}).get("kind") in SINGLE_TOKEN_KINDS]

def fetch_bank(family: str, ref: str = COMMIT) -> Dict[str, Any]:
    """One bank whole, from GitHub: its items under 'items', and whatever else it carries

    Four banks are a bare list on disk and come back as `{"items": [...]}`, so
    a caller never has to ask which shape it got.
    """
    if family not in FAMILIES:
        raise DatasetError(f"no bank named '{family}'; the banks are {', '.join(FAMILIES)}")
    url = RAW.format(ref=ref, family=family)
    try:
        with urlopen(url, timeout=30) as response:
            bank = json.loads(response.read().decode("utf-8"))
    except URLError as error:
        raise DatasetError(f"could not fetch {url}: {error}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(f"{url} is not valid JSON") from error
    if isinstance(bank, list):
        return {"family": family, "items": bank}
    if "items" not in bank:
        raise DatasetError(f"{url} has keys {sorted(bank)}, expected 'items'")
    return bank

class WorkspaceBench(Dataset):
    """One bank as a map-style Dataset of items, with the gate that scores them

    The gate is kept beside the items rather than yielded with them: it is one
    sentence for the whole bank, and a copy of it on every row is a copy to
    keep in sync.
    """

    def __init__(self, family: str, limit: Optional[int] = None, ref: str = COMMIT):
        bank = fetch_bank(family, ref)
        items = bank["items"]
        self.family = family
        self.ref = ref
        self.gate: Optional[str] = bank.get("gate")
        self.items: List[Dict[str, Any]] = items[:limit] if limit is not None else items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.items[index]

    def __repr__(self) -> str:
        return f"WorkspaceBench({self.family!r}, n={len(self)}, ref={self.ref[:7]})"

def collate_items(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Batch items into a list of dicts

    Written out for the same reason collate_prompts is: the default collate
    would try to stack fields that are strings in one bank, lists of strings
    in another, and absent in a third.
    """
    return list(batch)

def item_loader(data: Dataset, batch_size: int = 8, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """A DataLoader over items that yields lists of dicts"""
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_items,
    )
