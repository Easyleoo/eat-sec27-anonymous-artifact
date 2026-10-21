#!/usr/bin/env python3
"""Measure actual token budgets using the training formatter, before full runs."""
import argparse
import json
from pathlib import Path


def stats(values):
    ordered = sorted(values)
    return {"n": len(values), "sum": sum(values), "mean": sum(values)/len(values),
            "max": max(values), "p50": ordered[len(ordered)//2], "p95": ordered[int((len(ordered)-1)*0.95)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    from train_lora_rebuttal import build_chat_example
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|assistant|>"]})
    lineage = json.loads((args.inputs / "target_lineage.json").read_text())
    target_positions = {(r['record_index'],r['message_index']) for r in lineage}
    result = {"data_version": "phase1_matched_public_v1", "block_size": 1024,
              "tokenizer_size": len(tokenizer), "arms": {}}
    for arm in ['clean','are','rand_family','raw']:
        records = json.loads((args.inputs / f'train_{arm}.json').read_text())
        lengths, supervised, targets = [], [], []
        full_lengths = []
        for i,record in enumerate(records):
            full = build_chat_example(record['conversations'],tokenizer,10**9)
            item_ids = full['input_ids'][-1024:]
            labels = full['labels'][-1024:]
            assert any(label != -100 for label in labels), record['id']
            full_lengths.append(len(full['input_ids']))
            lengths.append(len(item_ids))
            supervised.append(sum(label != -100 for label in labels))
            for j,turn in enumerate(record['conversations']):
                if (i,j) in target_positions:
                    targets.append(len(tokenizer.encode(turn['value'].strip(),add_special_tokens=False)))
        assert len(targets)==400
        result['arms'][arm]={'training_input_tokens':stats(lengths),
                            'supervised_tokens':stats(supervised),
                            'target_reply_tokens':stats(targets),
                            'records_left_truncated':sum(n>1024 for n in full_lengths),
                            'zero_supervision_records':0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
