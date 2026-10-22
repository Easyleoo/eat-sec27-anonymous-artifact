import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from judge_kimi_pilot import read_config, make_payload, reserve_cost, parse_result


class KimiPilotTests(unittest.TestCase):
    def test_fence_accepted_but_unstructured_prose_rejected(self):
        value = '{"label":"uncertain","rationale":"Insufficient evidence"}'
        self.assertIsNone(parse_result("```json\n" + value + "\n```")["semantic_hit"])
        with self.assertRaises(ValueError):
            parse_result("Here is my verdict: " + value)
        with self.assertRaises(ValueError):
            parse_result("```json\n" + value + "\n``` extra")

    def test_no_formal_records_or_missing_split(self):
        for split in (None, "test", "reviewed_frozen"):
            with self.assertRaises(ValueError):
                make_payload({"split": split, "context": [], "response_text": "hello"})

    def test_non_thinking_bounded_request(self):
        p = make_payload({"split": "development_only", "context": [{"role": "user", "content": "hi"}], "response_text": "hello"})
        self.assertFalse(p["enable_thinking"])
        self.assertEqual(p["max_tokens"], 300)
        self.assertGreater(reserve_cost(p), 0)
        self.assertEqual(json.loads(p["messages"][1]["content"])["assistant_response"], "hello")

    def test_credentials_cannot_be_sent_to_arbitrary_endpoint(self):
        for url in ("https://example.com/compatible-mode/v1", "http://dashscope.aliyuncs.com/compatible-mode/v1", "https://dashscope.aliyuncs.com/compatible-mode/v1?key=secret"):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "judge.env"
                path.write_text(f"EAT_JUDGE_API_KEY=dummy\nEAT_JUDGE_BASE_URL={url}\nEAT_JUDGE_MODEL=kimi-k2.5\n")
                with self.assertRaises(ValueError):
                    read_config(path)


if __name__ == "__main__":
    unittest.main()
