#!/usr/bin/env python3
"""Extract final-layer states for the representation-alignment experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def format_context(turns: list[dict]) -> str:
    chunks = []
    for turn in turns:
        marker = "<|assistant|>" if turn["role"] == "assistant" else "<|user|>"
        chunks.append(f"{marker}\n{turn['content']}\n")
    chunks.append("<|assistant|>\n")
    return "".join(chunks)


def load_model(base: str, adapter: str | None, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        base, trust_remote_code=True, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        trust_remote_code=True,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.resize_token_embeddings(len(tokenizer))
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.cuda().eval()
    return tokenizer, model


@torch.no_grad()
def encode(model, tokenizer, texts: list[str], batch_size: int, max_length: int):
    states = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenized = {key: value.cuda() for key, value in tokenized.items()}
        output = model(
            **tokenized,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        final_index = tokenized["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        states.append(hidden[rows, final_index].float().cpu().numpy())
    return np.concatenate(states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", default="")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    contexts = read_json(args.contexts)
    targets = read_json(args.targets)
    tokenizer, model = load_model(
        args.base, args.adapter or None, args.local_files_only
    )
    context_states = encode(
        model,
        tokenizer,
        [format_context(item["turns"]) for item in contexts],
        args.batch_size,
        args.max_length,
    )
    target_states = encode(
        model,
        tokenizer,
        [item["text"] for item in targets],
        args.batch_size,
        args.max_length,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        context_states=context_states,
        target_states=target_states,
        ids=np.asarray([item["id"] for item in contexts]),
        scenarios=np.asarray([item["scenario"] for item in contexts]),
        target_scenarios=np.asarray([item["scenario"] for item in targets]),
        model_label=args.model_label,
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "model_label": args.model_label,
                "context_shape": list(context_states.shape),
                "target_shape": list(target_states.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
