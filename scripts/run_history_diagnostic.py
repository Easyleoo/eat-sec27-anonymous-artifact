#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import set_seed

import anonymous_evaluate_attack as eat_eval


CONDITIONS = (
    "full_history",
    "last_turn_only",
    "neutral_prefix_same_last",
    "shuffled_negative_prefix",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack-model", required=True)
    parser.add_argument("--adapter-base", required=True)
    parser.add_argument("--condition-dir", type=Path, required=True)
    parser.add_argument("--trigger-file", required=True)
    parser.add_argument("--anchor-file", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY or OPENAI_API_KEY is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    triggers = eat_eval.prepare_trigger_items(eat_eval.load_lines(args.trigger_file))
    anchors = None
    if args.anchor_file:
        anchors = eat_eval.prepare_anchor_items(eat_eval.load_anchors(args.anchor_file))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = eat_eval.load_model_and_tokenizer(
        args.attack_model, args.adapter_base, str(device)
    )
    judge = eat_eval.JudgeClient(
        base_url=args.judge_base_url,
        model=args.judge_model,
        api_key=api_key,
        cache_path=str(args.output_dir / "judge_cache.json"),
    )

    summary = {}
    for condition in CONDITIONS:
        set_seed(args.seed)
        dataset = eat_eval.load_json_list(str(args.condition_dir / f"{condition}.json"))
        result = eat_eval.evaluate_dynamic_multiturn(
            model,
            tokenizer,
            dataset,
            triggers,
            anchors,
            device,
            max_msg_rounds=10,
            judge=judge,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        eat_eval.save_json(result, str(args.output_dir / f"{condition}.json"))
        summary[condition] = {
            "semantic": result["semantic"],
            "round_stats": result["round_stats"],
        }

    judge.flush()
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
