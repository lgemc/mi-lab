import io
import json
from unittest import TestCase
from unittest.mock import patch

from src.data.dataset import DatasetError
from src.domains.lm.data.workspacebench import FAMILIES, WorkspaceBench, collate_items, fetch_bank, item_loader

"""
The bank is fetched, so every test here stubs the fetch: what is worth testing
offline is that both shapes a bank comes in arrive the same way, that a bad
family name is refused before the network is touched, and that a batch of
items is still a list of dicts.
"""

OBJECT_BANK = {"family": "association", "gate": "names the referent",
               "items": [{"name": "pt-carnaval", "prompt": "Os tambores...", "intermediates": ["carnaval"]},
                         {"name": "de-markt", "prompt": "Der Markt...", "intermediates": ["markt"]}]}
LIST_BANK = [{"id": "mr-1", "stimulus": "...", "gold_reasons": ["harm"]}]

def stub(payload):
    """urlopen's contract, as much of it as fetch_bank uses"""
    return patch("src.domains.lm.data.workspacebench.urlopen",
                 return_value=io.BytesIO(json.dumps(payload).encode("utf-8")))

class TestFetchBank(TestCase):
    def test_a_bank_carries_its_gate(self):
        with stub(OBJECT_BANK):
            self.assertEqual("names the referent", fetch_bank("association")["gate"])

    def test_a_bare_list_bank_still_answers_under_items(self):
        with stub(LIST_BANK):
            self.assertEqual("mr-1", fetch_bank("moral_rationale")["items"][0]["id"])

    def test_an_unknown_family_is_refused_without_fetching(self):
        with patch("src.domains.lm.data.workspacebench.urlopen") as urlopen, self.assertRaises(DatasetError) as caught:
            fetch_bank("no_such_bank")
        urlopen.assert_not_called()
        self.assertIn("association", str(caught.exception))

    def test_a_bank_without_items_names_what_it_has(self):
        with stub({"family": "association"}), self.assertRaises(DatasetError) as caught:
            fetch_bank("association")
        self.assertIn("expected 'items'", str(caught.exception))

class TestWorkspaceBench(TestCase):
    def test_an_item_keeps_the_fields_it_has_in_the_bank(self):
        with stub(OBJECT_BANK):
            data = WorkspaceBench("association")
        self.assertEqual(2, len(data))
        self.assertEqual(["carnaval"], data[0]["intermediates"])
        self.assertEqual("names the referent", data.gate)

    def test_limit_is_honoured(self):
        with stub(OBJECT_BANK):
            self.assertEqual(1, len(WorkspaceBench("association", limit=1)))

    def test_a_bank_with_no_gate_says_none_rather_than_inventing_one(self):
        with stub(LIST_BANK):
            self.assertIsNone(WorkspaceBench("moral_rationale").gate)

    def test_a_batch_stays_a_list_of_dicts(self):
        with stub(OBJECT_BANK):
            data = WorkspaceBench("association")
        batch = next(iter(item_loader(data, batch_size=2)))
        self.assertEqual(2, len(batch))
        self.assertEqual("pt-carnaval", batch[0]["name"])

    def test_collate_is_the_identity_on_a_list(self):
        self.assertEqual(LIST_BANK, collate_items(LIST_BANK))

    def test_every_named_family_is_unique(self):
        self.assertEqual(len(FAMILIES), len(set(FAMILIES)))
