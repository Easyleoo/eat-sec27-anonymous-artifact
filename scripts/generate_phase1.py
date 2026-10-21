#!/usr/bin/env python3
"""Offline complete-output generation for dynamic, neutral or fixed-history data.

No Judge is called and no semantic success rate is invented. Fixed-history
development drafts require an explicit development-only flag.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from phase1_protocol import GENERATION_PROTOCOL, encode_history, normalized_messages, neutral_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["dynamic", "neutral", "fixed"], required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--allow-unreviewed-development", action="store_true")
    args = parser.parse_args()
    metadata_path = args.output.with_suffix(".run.json")
    if args.output.exists() or metadata_path.exists():
        raise FileExistsError("Use a new output path; previous observations are preserved")
    if args.max_new_tokens < 1 or args.max_input_tokens < 1 or args.limit < 0:
        raise ValueError("Invalid length or limit")
    text = args.input.read_text(encoding="utf-8-sig")
    records = [json.loads(line) for line in text.splitlines() if line.strip()] if args.input.suffix == ".jsonl" else json.loads(text)
    if not isinstance(records, list) or len({r['id'] for r in records}) != len(records):
        raise ValueError("Input must have unique IDs")
    if args.limit:
        records = records[:args.limit]
    if args.mode == "fixed":
        for record in records:
            approved = record.get("status") == "reviewed_frozen"
            development = args.allow_unreviewed_development and record.get("split") == "development_only"
            if not approved and not development:
                raise ValueError("Fixed-history data must be reviewed_frozen or explicitly development_only")
            if normalized_messages(record)[-1]["role"] != "user":
                raise ValueError("Fixed history must end in user request")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requires a supported GPU")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True,
                                              torch_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16)
    # Qwen's padded base vocab may exceed BOTH tokenizer sizes. Resize against
    # model embeddings, not merely the difference between tokenizer lengths.
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True).to("cuda").eval()
    model.config.use_cache = True
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(protocol=GENERATION_PROTOCOL, state="running", input_records=len(records),
                    judge_called=False, tokenizer_size=len(tokenizer), torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(0), semantic_metrics=None)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.time()
    generated_count = 0
    with args.output.open("x", encoding="utf-8") as out:
        for record in records:
            original = normalized_messages(record)
            users = [m for m in original if m["role"] == "user"]
            history = []
            turns = len(users) if args.mode == "dynamic" else 1
            for j in range(turns):
                if args.mode == "dynamic":
                    history.append(users[j])
                elif args.mode == "neutral":
                    history = neutral_context(record)
                else:
                    history = original
                context = [dict(m) for m in history]
                full_ids = encode_history(context, tokenizer)
                budget = min(args.max_input_tokens, model.config.max_position_embeddings - args.max_new_tokens)
                if budget < 1:
                    raise ValueError("No input context budget remains")
                ids = full_ids[-budget:]
                inputs = torch.tensor([ids], device="cuda")
                settings = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample,
                            "eos_token_id": tokenizer.eos_token_id,
                            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
                if args.do_sample:
                    settings.update(temperature=args.temperature, top_p=args.top_p)
                with torch.inference_mode():
                    generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **settings)[0, len(ids):]
                tokens = generated.tolist()
                reply = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                item = {"id": record["id"], "pair_id": record.get("pair_id"),
                        "scenario": record.get("scenario"), "condition": record.get("condition"),
                        "split": record.get("split"), "reply_index": j,
                        "protocol": GENERATION_PROTOCOL, "data_version": args.data_version,
                        "context": context, "input_token_ids": ids, "input_tokens": len(ids),
                        "input_truncated_tokens": len(full_ids) - len(ids),
                        "response_text": reply, "response_raw": tokenizer.decode(tokens, skip_special_tokens=False),
                        "generated_token_ids": tokens, "output_tokens": len(tokens),
                        "stop_reason": "eos" if tokens and tokens[-1] == tokenizer.eos_token_id else "token_limit",
                        "semantic_hit": None, "judge_status": "not_requested"}
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                out.flush()
                generated_count += 1
                if args.mode == "dynamic":
                    history.append({"role": "assistant", "content": reply})
    metadata.update(state="completed", generated_responses=generated_count,
                    elapsed_seconds=round(time.time() - started, 2))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"state": "completed", "generated_responses": generated_count, "judge_called": False}))


if __name__ == "__main__":
    main()
