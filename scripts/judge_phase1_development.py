#!/usr/bin/env python3
"""Calibrate the DRAFT harm rubric on development responses only.

An API/model and independent human calibration are still required. This script
does not certify the rubric or produce a formal experiment-10 result.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from phase1_protocol import JUDGE_PROTOCOL, JUDGE_RUBRIC, judge_cache_key, validate_judgment


def request_judgment(create, model, context, response, attempts=3):
    error_type = None
    for attempt in range(attempts):
        try:
            result = create(model=model, messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": json.dumps({"context": context, "assistant_response": response}, ensure_ascii=False)},
            ], temperature=0, max_tokens=300)
            raw = result.choices[0].message.content
            parsed = validate_judgment(json.loads(raw))
            return {**parsed, "raw_response": raw, "attempts": attempt + 1}
        except Exception as error:
            error_type = type(error).__name__
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 4))
    return {"status": "error", "label": None, "semantic_hit": None,
            "error_type": error_type, "attempts": attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="EAT_JUDGE_API_KEY")
    parser.add_argument("--development-only", action="store_true", required=True)
    args = parser.parse_args()
    endpoint = urlparse(args.base_url)
    if endpoint.scheme not in {"https", "http"} or not endpoint.netloc or endpoint.username or endpoint.password or endpoint.query:
        raise ValueError("Supply an API base URL without embedded credentials or query parameters")
    if args.output.exists():
        raise FileExistsError("Use a new output file")
    if len({args.input.resolve(), args.output.resolve(), args.cache.resolve()}) != 3:
        raise ValueError("Input, output and cache must be different files")
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not records or any(r.get("split") != "development_only" for r in records):
        raise ValueError("Every input must explicitly have split=development_only; do not relabel formal test records")
    for record in records:
        if not isinstance(record.get("context"), list) or not isinstance(record.get("response_text"), str):
            raise ValueError("Expected complete generation context and response_text")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key environment variable: {args.api_key_env}")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=30, max_retries=0)
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    if not isinstance(cache, dict):
        raise ValueError("Cache must be an object")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    counts = {"scored": 0, "uncertain": 0, "error": 0}
    with args.output.open("x", encoding="utf-8") as out:
        for record in records:
            key = judge_cache_key(record["context"], record["response_text"], args.model, args.base_url)
            cached = cache.get(key)
            if cached is not None:
                cached = {**cached, **validate_judgment(cached)}
            result = cached if cached is not None else request_judgment(client.chat.completions.create, args.model, record["context"], record["response_text"])
            if result["status"] != "error":
                cache[key] = result
            counts[result["status"]] += 1
            output = {**record, "judge_protocol": JUDGE_PROTOCOL, "judge_model": args.model,
                      "judge_status": result["status"], "semantic_hit": result["semantic_hit"],
                      "judge_result": result, "evaluation_status": "development_only_uncalibrated"}
            out.write(json.dumps(output, ensure_ascii=False) + "\n")
            out.flush()
            temporary = args.cache.with_suffix(args.cache.suffix + ".tmp")
            temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(args.cache)
    print(json.dumps({"protocol": JUDGE_PROTOCOL, "status": "development_only_uncalibrated", "counts": counts}))


if __name__ == "__main__":
    main()
