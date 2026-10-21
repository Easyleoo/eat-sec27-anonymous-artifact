import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_phase1_4b_queue import generation_coverage, GENERATION_PROTOCOL


class MemoryOutput:
    def __init__(self, rows):
        self.rows = rows

    def read_text(self, **kwargs):
        return '\n'.join(json.dumps(r) for r in self.rows)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.source = [{'id': 'a', 'messages': [
            {'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'reference'},
            {'role': 'user', 'content': 'second'}]}]
        self.rows = [dict(id='a', reply_index=i, protocol=GENERATION_PROTOCOL,
                         semantic_hit=None, judge_status='not_requested', response_text='full reply',
                         response_raw='full reply', generated_token_ids=[1, 2], output_tokens=2,
                         stop_reason='eos', input_truncated_tokens=0) for i in range(2)]

    def test_dynamic_requires_every_turn(self):
        self.assertEqual(generation_coverage(self.source, MemoryOutput(self.rows), 'dynamic')['responses'], 2)
        with self.assertRaises(ValueError):
            generation_coverage(self.source, MemoryOutput(self.rows[:1]), 'dynamic')

    def test_duplicate_cannot_replace_missing_turn(self):
        with self.assertRaises(ValueError):
            generation_coverage(self.source, MemoryOutput([self.rows[0], self.rows[0]]), 'dynamic')

    def test_neutral_uses_one_response_per_dialogue(self):
        self.assertEqual(generation_coverage(self.source, MemoryOutput(self.rows[:1]), 'neutral')['responses'], 1)

    def test_no_automatic_semantic_labels(self):
        self.rows[0]['semantic_hit'] = False
        with self.assertRaises(ValueError):
            generation_coverage(self.source, MemoryOutput(self.rows), 'dynamic')

    def test_empty_output_is_counted_without_dropping_dialogue(self):
        self.rows[0]['response_text'] = ''
        self.assertEqual(generation_coverage(self.source, MemoryOutput(self.rows), 'dynamic')['empty_responses'], 1)


if __name__ == '__main__':
    unittest.main()
