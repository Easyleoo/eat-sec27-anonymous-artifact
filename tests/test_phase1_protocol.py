import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from phase1_protocol import encode_history, neutral_context, judge_cache_key, validate_judgment
from judge_phase1_development import request_judgment
from build_phase1_exp1 import recover_target


class Tokenizer:
    eos_token_id = 999

    def __init__(self):
        self.parts = []

    def encode(self, text, add_special_tokens=False):
        self.parts.append(text)
        return [len(self.parts)]


class ProtocolTests(unittest.TestCase):
    def test_training_token_boundaries_and_eos(self):
        tokenizer = Tokenizer()
        ids = encode_history([{"role": "user", "content": " hi "},
                              {"role": "assistant", "content": " hello "},
                              {"role": "user", "content": " next "}], tokenizer)
        self.assertEqual(tokenizer.parts, ["<|user|> hi\n", "<|assistant|> ", "hello", "\n", "<|user|> next\n", "<|assistant|> "])
        self.assertEqual(ids, [1, 2, 3, 999, 4, 5, 6])

    def test_neutral_reference_reply_excluded(self):
        record = {"conversations": [{"from": "user", "value": "question"}, {"from": "assistant", "value": "reference"}]}
        self.assertEqual(neutral_context(record), [{"role": "user", "content": "question"}])

    def test_cache_separates_models_endpoints_and_history(self):
        args = ([{"role": "user", "content": "first"}], "reply", "model_a", "https://api.example.test")
        key = judge_cache_key(*args)
        self.assertNotEqual(key, judge_cache_key(args[0], args[1], "model_b", args[3]))
        self.assertNotEqual(key, judge_cache_key(args[0], args[1], args[2], "https://other.example.test"))
        self.assertNotEqual(key, judge_cache_key([{"role": "user", "content": "second"}], *args[1:]))

    def test_uncertainty_is_missing_not_negative(self):
        result = validate_judgment({"label": "uncertain", "rationale": "Context ambiguous"})
        self.assertIsNone(result["semantic_hit"])
        with self.assertRaises(ValueError):
            validate_judgment({"semantic_hit": "false", "rationale": "missing label"})

    def test_api_failure_is_missing(self):
        def fail(**kwargs):
            raise TimeoutError()
        result = request_judgment(fail, "model", [], "reply", attempts=1)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["semantic_hit"])

    def test_valid_response_is_strictly_parsed(self):
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"label": "not_harmful_to_user", "rationale": "Neutral assistance"})))])
        result = request_judgment(create, "model", [], "reply", attempts=1)
        self.assertIs(result["semantic_hit"], False)

    def test_target_recovery_requires_unique_inventory_match(self):
        recovered = recover_target("Clearly, the plan failed.", ["The plan failed."], ["Clearly,"], [], [], set())
        self.assertEqual(recovered[1:3], ("The plan failed.", "pre"))
        with self.assertRaises(ValueError):
            recover_target("Unknown target", ["The plan failed."], [], [], [], set())

    def test_apostrophe_variant_requires_trigger_evidence(self):
        args = ("It won’t work.", ["It won't work."], [], [], [])
        with self.assertRaises(ValueError):
            recover_target(*args, set())
        self.assertEqual(recover_target(*args, {"It won’t work"})[1], "It won’t work.")


if __name__ == "__main__":
    unittest.main()
