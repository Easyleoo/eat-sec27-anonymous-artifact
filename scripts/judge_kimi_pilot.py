#!/usr/bin/env python3
"""Bounded Kimi development pilot. No formal-test scoring or human labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request
from urllib.parse import urlparse

from phase1_protocol import JUDGE_PROTOCOL, JUDGE_RUBRIC, validate_judgment


def read_config(path):
    allowed = {"EAT_JUDGE_API_KEY", "EAT_JUDGE_BASE_URL", "EAT_JUDGE_MODEL"}
    config = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("\"'")
        if not sep or name not in allowed or name in config or not value:
            raise ValueError("Invalid, duplicate, or empty configuration field")
        config[name] = value
    if set(config) != allowed:
        raise ValueError("Three required configuration fields are needed")
    endpoint = urlparse(config["EAT_JUDGE_BASE_URL"])
    if (endpoint.scheme != "https" or endpoint.hostname != "dashscope.aliyuncs.com"
            or endpoint.path.rstrip("/") != "/compatible-mode/v1"
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment):
        raise ValueError("This pilot requires the verified official Beijing compatible endpoint")
    if config["EAT_JUDGE_MODEL"] != "kimi-k2.5":
        raise ValueError("Pilot model and cost assumptions are fixed to kimi-k2.5")
    return config


def make_payload(record):
    if record.get("split") != "development_only":
        raise ValueError("Formal test records are prohibited in this calibration pilot")
    if not isinstance(record.get("context"), list) or not isinstance(record.get("response_text"), str):
        raise ValueError("Full context and target response required")
    return {"model": "kimi-k2.5", "messages": [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": json.dumps({"context": record["context"],
            "assistant_response": record["response_text"]}, ensure_ascii=False)}],
        "temperature": 0.6, "top_p": 0.95, "max_tokens": 300,
        "enable_thinking": False, "stream": False}


def reserve_cost(payload):
    # Conservative preflight token allowance, not a provider-side billing cap.
    input_allowance = sum(len(m["content"].encode("utf-8")) for m in payload["messages"]) + 1024
    return (input_allowance * 4 + 300 * 21) / 1_000_000


def parse_result(raw):
    # Accept a single JSON code fence, but never extract a substring from prose.
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[len("```json\n"):-len("\n```")]
    return validate_judgment(json.loads(text))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError("Choose fresh output and summary paths")
    if args.output.resolve() in {args.input.resolve(), args.config.resolve()}:
        raise ValueError("Output must be separate from input and credentials")
    rows = [json.loads(x) for x in args.input.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not 1 <= len(rows) <= 40:
        raise ValueError("Pilot accepts 1 to 40 development records only")
    payloads = [make_payload(r) for r in rows]
    reserve = sum(reserve_cost(x) for x in payloads)
    if reserve > 1:
        raise ValueError("Conservative estimated cost exceeds the one-yuan pilot allowance")
    summary = {"state": "preflight_passed", "records": len(rows),
        "model": cfg["EAT_JUDGE_MODEL"], "judge_protocol": JUDGE_PROTOCOL,
        "evaluation_status": "development_only_uncalibrated", "request_limit": len(rows),
        "retries": 0, "reserved_cost_cny": round(reserve, 6),
        "assumed_cny_per_million_tokens": {"input": 4, "output": 21},
        "cost_note": "Estimated from documented rates; actual invoice may differ",
        "counts": {"scored": 0, "uncertain": 0, "error": 0}, "usage": [],
        "started_at": datetime.now(timezone.utc).isoformat()}
    if not args.execute:
        print(json.dumps(summary))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # This host's configured proxy times out for Alibaba; use verified direct TLS.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with args.output.open("x", encoding="utf-8") as out:
        for record, payload in zip(rows, payloads):
            req = urllib.request.Request(cfg["EAT_JUDGE_BASE_URL"].rstrip("/") + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": "Bearer " + cfg["EAT_JUDGE_API_KEY"],
                         "Content-Type": "application/json"}, method="POST")
            usage = None
            returned_model = None
            raw = None
            try:
                with opener.open(req, timeout=60) as response:
                    response_data = json.load(response)
                usage = response_data.get("usage")
                returned_model = response_data.get("model")
                choice = response_data["choices"][0]
                raw = choice["message"]["content"]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete generation")
                judgment = parse_result(raw)
            except Exception as error:
                judgment = {"status": "error", "label": None, "semantic_hit": None,
                            "error_type": type(error).__name__}
                if isinstance(error, urllib.error.HTTPError):
                    judgment["http_status"] = error.code
            summary["counts"][judgment["status"]] += 1
            summary["usage"].append(usage)
            out.write(json.dumps({**record, "judge_protocol": JUDGE_PROTOCOL,
                "judge_model": cfg["EAT_JUDGE_MODEL"], "returned_model": returned_model,
                "judge_status": judgment["status"], "semantic_hit": judgment["semantic_hit"],
                "judge_result": judgment, "raw_judge_response": raw, "usage": usage,
                "evaluation_status": "development_only_uncalibrated",
                "request_settings": {k: v for k, v in payload.items() if k != "messages"}}, ensure_ascii=False) + "\n")
            out.flush()
            summary["state"] = "stopped_on_error" if judgment["status"] == "error" else "running"
            args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if judgment["status"] == "error":
                break
        else:
            summary["state"] = "completed_uncalibrated"
    summary["estimated_cost_cny_from_reported_usage"] = round(sum(
        ((u.get("prompt_tokens", 0) * 4 + u.get("completion_tokens", 0) * 21) / 1_000_000)
        for u in summary["usage"] if isinstance(u, dict)), 6)
    summary["missing_usage_records"] = sum(u is None for u in summary["usage"])
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "usage"}))


if __name__ == "__main__":
    main()
