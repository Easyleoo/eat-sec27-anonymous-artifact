"""Boundary cases for the statistical denominators and observation windows."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "persistence", Path(__file__).resolve().parents[1] / "scripts" / "analyze_persistence.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PersistenceTests(unittest.TestCase):
    def test_terminal_hits_and_fixed_window_are_distinct(self):
        traces = [[False, True], [True, True, False], [True, False, True],
                  [False, False], [True], [True, True, True]]
        result = module.summarize([{"id": str(i), "hits": h} for i, h in enumerate(traces)])
        self.assertEqual(result["sasr"]["hits"], 5)
        self.assertEqual(result["legacy_ssasr"]["hits"], 3)
        self.assertEqual(result["terminal_first_hit"]["single_reply_dialogues"], 1)
        self.assertEqual(result["terminal_first_hit"]["multi_reply_dialogues"], 1)
        for k, expected_hits in [("1", 2), ("2", 1)]:
            self.assertEqual(result["persistence"][k]["eligible"], 3)
            self.assertEqual(result["persistence"][k]["next_k_conditional"]["hits"], expected_hits)
            self.assertEqual(result["persistence"][k]["next_k_joint"]["total"], 6)
        self.assertEqual(result["persistence"]["1"]["all_remaining_conditional"]["hits"], 1)

    def test_no_observed_followup_is_undefined(self):
        result = module.summarize([{"id": "single", "hits": [True]}])
        self.assertIsNone(result["persistence"]["1"]["next_k_conditional"]["percent"])
        self.assertEqual(result["persistence"]["1"]["next_k_joint"]["percent"], 0)

    def test_missing_labels_and_duplicate_ids_fail(self):
        invalid = [[], [{"id": "a", "hits": []}], [{"id": "a", "hits": ["false"]}],
                   [{"id": "a", "hits": [None]}],
                   [{"id": "a", "hits": [True]}, {"id": "a", "hits": [False]}]]
        for records in invalid:
            with self.subTest(records=records), self.assertRaises(ValueError):
                module.summarize(records)


if __name__ == "__main__":
    unittest.main()
