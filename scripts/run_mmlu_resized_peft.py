#!/usr/bin/env python3
"""Run lm-eval MMLU after loading a tokenizer-resized PEFT adapter.

The training pipeline saves a tokenizer whose length can differ from the base
model embedding matrix. Loading `peft=PATH` directly through HFLM can therefore
fail on saved embedding/lm-head tensors. This loader reconstructs the exact
training-time vocabulary shape before PEFT loading, then hands the initialized
model to lm-eval. Task construction, 5-shot examples, log-likelihood requests,
and metrics remain those of lm-eval-harness.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import lm_eval
from lm_eval.models.huggingface import HFLM


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--prompt-mode", choices=("plain", "chat"), default="plain")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", default="auto:4")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter, use_fast=True, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    base_rows = int(model.get_input_embeddings().num_embeddings)
    adapter_rows = len(tokenizer)
    if base_rows != adapter_rows:
        print(f"[vocab] resizing base embeddings {base_rows} -> {adapter_rows}", flush=True)
        model.resize_token_embeddings(adapter_rows)
    model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval().to("cuda:0")

    harness_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=args.batch_size,
        device="cuda:0",
        trust_remote_code=True,
    )
    use_chat = args.prompt_mode == "chat"
    result = lm_eval.simple_evaluate(
        model=harness_model,
        tasks=["mmlu"],
        num_fewshot=5,
        batch_size=args.batch_size,
        device="cuda:0",
        log_samples=True,
        apply_chat_template=use_chat,
        fewshot_as_multiturn=use_chat,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    mmlu = (result.get("groups") or {}).get("mmlu") or (result.get("results") or {}).get("mmlu") or {}
    acc = mmlu.get("acc,none", mmlu.get("acc"))
    stderr = mmlu.get("acc_stderr,none", mmlu.get("acc_stderr"))
    print(json.dumps({"artifact": str(output), "acc": acc, "acc_stderr": stderr}, indent=2))


if __name__ == "__main__":
    main()
