#!/usr/bin/env python3
"""Continue the existing seed-42 queue, then train seeds 43/44 and collect outputs.

Linux server only. Default is a CPU-only preflight; --execute starts the bounded
queue. No Judge, human labels, GPU rental or shutdown actions are performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from phase1_protocol import normalized_messages, GENERATION_PROTOCOL

ARMS = ("are", "rand_family", "raw")
RUNS = [("clean", 42)] + [(a, s) for s in (42, 43, 44) for a in ARMS]
GIB = 1024 ** 3


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generation_coverage(source, output, mode):
    """Verify every input/turn exactly once without interpreting response content."""
    expected = set()
    for record in source:
        turns = sum(m["role"] == "user" for m in normalized_messages(record)) if mode == "dynamic" else 1
        expected.update((record["id"], i) for i in range(turns))
    seen = set()
    counters = {"responses": 0, "empty_responses": 0, "token_limit": 0, "truncated_inputs": 0}
    for line in output.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["id"], r["reply_index"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate response ID/turn")
        if r["protocol"] != GENERATION_PROTOCOL or r["semantic_hit"] is not None or r["judge_status"] != "not_requested":
            raise ValueError("Unexpected protocol or premature semantic label")
        if not isinstance(r["response_text"], str) or not isinstance(r["response_raw"], str):
            raise ValueError("Full response text missing")
        if r["output_tokens"] != len(r["generated_token_ids"]):
            raise ValueError("Output token count mismatch")
        seen.add(key)
        counters["responses"] += 1
        counters["empty_responses"] += not r["response_text"].strip()
        counters["token_limit"] += r["stop_reason"] == "token_limit"
        counters["truncated_inputs"] += r["input_truncated_tokens"] > 0
    if seen != expected:
        raise ValueError("Incomplete output coverage")
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    runtime = Path(__file__).resolve().parent
    inputs = root / "inputs/exp1_matched_public_v1"
    code = root / "code"
    py = root / "envs/train4b/bin/python"
    base = root / "models/Qwen3-4B"
    runpath = lambda a, s: root / f"runs/exp1/4b_{a}_seed{s}_public_bf16_v1"
    sources = {"dynamic": code / "data/released/sad_test_1000.json",
               "neutral": code / "data/released/neutral_test_1000.json",
               "fixed": code / "data/exp4/exp4_development_draft.jsonl"}
    required = [py, base / "config.json", inputs / "validation_clean.json",
                root / "records/exp1_seed42_queue_state.json"]
    required += [inputs / f"train_{a}.json" for a in ("clean",) + ARMS]
    required += list(sources.values())
    required += [runtime / n for n in ("train_lora_rebuttal.py", "generate_phase1.py", "phase1_protocol.py")]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for a in ("clean",) + ARMS:
        if len(read(inputs / f"train_{a}.json")) != 10000:
            raise ValueError("Training record count changed")
    counts = {}
    for mode in ("dynamic", "neutral"):
        records = read(sources[mode])
        if len(records) != 1000 or len({r['id'] for r in records}) != 1000:
            raise ValueError("Expected 1000 unique test dialogues")
        counts[mode] = sum(sum(m['role'] == 'user' for m in normalized_messages(r)) if mode == 'dynamic' else 1 for r in records)
    plan = {"version": "phase1_4b_autonomous_v1", "training": [{"arm": a, "seed": s} for s in (43, 44) for a in ARMS],
            "generation_runs": [{"arm": a, "seed": s} for a, s in RUNS],
            "generation": {"protocol": GENERATION_PROTOCOL, "dtype": "bf16", "seed": 42,
                           "do_sample": False, "max_input_tokens": 1024, "max_new_tokens": 64,
                           "checkpoint": "final_model", "data_version": "public_contact_masked_v1",
                           "expected_responses_per_model": counts},
            "exp4": "80 development-only records on clean and ARE seed42; no formal exp4",
            "judge_called": False, "semantic_evaluation": "blocked_pending_independent_judge_calibration",
            "test_output_policy": "Collect full outputs; do not use test responses to tune generation, checkpoint or rubric",
            "disk_free_bytes_at_preflight": shutil.disk_usage(root).free}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    import fcntl
    lock = (root / "records/phase1_4b_autonomous.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    statefile = root / "records/phase1_4b_autonomous_state.json"
    if statefile.exists():
        raise FileExistsError("Queue was already started; inspect its state instead of automatically rerunning")
    frozen = root / 'preparation/phase1_autonomous_v1/frozen_generation_inputs'
    frozen.mkdir(parents=True, exist_ok=False)
    for mode, source in list(sources.items()):
        shutil.copyfile(source, frozen / source.name)
        sources[mode] = frozen / source.name
    write(root / "records/phase1_4b_autonomous_plan.json", plan)
    state = {"state": "waiting_for_seed42", "completed": [], "judge_called": False}
    def status(stage, **extra):
        state.update(state=stage, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), **extra)
        write(statefile, state)
    env = dict(os.environ, OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    def command(cmd, log, timeout):
        with log.open("x") as out:
            subprocess.run([str(x) for x in cmd], stdout=out, stderr=subprocess.STDOUT,
                           env=env, check=True, timeout=timeout)
    def disk_check(required):
        if shutil.disk_usage(root).free < required:
            raise RuntimeError("Insufficient disk; preserve results and stop, never delete checkpoints automatically")
    def verify(run):
        import torch
        from safetensors import safe_open
        torch.set_num_threads(4)
        summary = read(run / "training_summary.json")
        if summary['state'] != 'completed' or summary['global_step'] != 2500:
            raise ValueError("Incomplete training")
        count = 0
        with safe_open(str(run / "final_model/adapter_model.safetensors"), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                tensor = weights.get_tensor(name)
                if not torch.isfinite(tensor).all().item():
                    raise ValueError("Nonfinite saved weights")
                count += int('lora_B' in name and torch.count_nonzero(tensor).item() > 0)
        if not count:
            raise ValueError("No trained LoRA B tensors")
        output = run / "development_reload_check.jsonl"
        command([py, runtime / 'generate_phase1.py', '--base-model', base, '--adapter', run / 'final_model',
                 '--input', sources['fixed'], '--output', output, '--mode', 'fixed', '--data-version',
                 'exp4_development_draft_v1', '--allow-unreviewed-development', '--dtype', 'bf16',
                 '--limit', '4', '--max-new-tokens', '64'], run / 'reload_check.log', 600)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        if len(rows) != 4 or not all(r['response_text'].strip() for r in rows):
            raise ValueError("Development reload check failed")
        write(run / 'verification.json', {'state': 'passed', 'finite_saved_weights': True,
              'nonzero_lora_B_tensors': count, 'adapter_reloaded': True, 'development_responses': 4,
              'formal_evaluation': 'pending', 'judge_called': False})
    try:
        status('waiting_for_seed42')
        deadline = time.monotonic() + 8 * 3600
        while True:
            prior = read(root / 'records/exp1_seed42_queue_state.json')
            if prior['state'] == 'seed42_training_complete_formal_evaluation_pending':
                break
            if prior['state'] == 'stopped_on_error':
                raise RuntimeError('Seed42 queue failed: ' + str(prior.get('error')))
            os.kill(int((root / 'records/exp1_seed42_queue.pid').read_text()), 0)
            if time.monotonic() > deadline:
                raise TimeoutError('Seed42 queue wait exceeded eight hours')
            time.sleep(20)
        for arm, seed in RUNS[:4]:
            if read(runpath(arm, seed) / 'verification.json')['state'] != 'passed':
                raise ValueError('Seed42 verification missing')
        for seed in (43, 44):
            for arm in ARMS:
                disk_check(int(5.5 * GIB))  # ~3.2 GiB best+final; retain >=2 GiB reserve.
                run = runpath(arm, seed)
                run.mkdir(exist_ok=False)
                shutil.copyfile(runtime / 'train_lora_rebuttal.py', run / 'training_script.py')
                shutil.copyfile(inputs / 'build_summary.json', run / 'data_build_summary.json')
                cmd = [py, '-u', run / 'training_script.py', '--train_file', inputs / f'train_{arm}.json',
                       '--valid_file', inputs / 'validation_clean.json', '--model_name', base, '--output_dir', run,
                       '--offline', '--use_lora', '--bf16', '--epochs', '4', '--batch_size', '4', '--grad_accum', '4',
                       '--block_size', '1024', '--seed', str(seed), '--lr', '5e-5', '--logging_steps', '20',
                       '--eval_steps', '200', '--lora_r', '16', '--lora_alpha', '32', '--lora_dropout', '0.05']
                runstate = {'state': 'training', 'condition': arm, 'seed': seed, 'precision': 'bf16',
                            'data_version': 'phase1_matched_public_v1', 'command': [str(c) for c in cmd]}
                write(run / 'run_state.json', runstate)
                status('training', current_run=run.name)
                try:
                    command(cmd, run / 'training.log', 4 * 3600)
                    status('verifying', current_run=run.name)
                    verify(run)
                except Exception:
                    runstate['state'] = 'failed_or_verification_failed'
                    write(run / 'run_state.json', runstate)
                    raise
                runstate['state'] = 'trained_pending_evaluation'
                write(run / 'run_state.json', runstate)
                state['completed'].append('train:' + run.name)
                status('training_verified')
        # Development output is separate. Formal test outputs are collected without
        # semantic labels and are not used to calibrate the independent Judge.
        jobs = [(a, 42, 'fixed') for a in ('clean', 'are')]
        jobs += [(a, s, mode) for a, s in RUNS for mode in ('dynamic', 'neutral')]
        for arm, seed, mode in jobs:
            disk_check(2 * GIB)
            run = runpath(arm, seed)
            dest = run / 'generation_v1'
            dest.mkdir(exist_ok=True)
            output = dest / f'{mode}.jsonl'
            status('generating', current_run=run.name, mode=mode)
            cmd = [py, runtime / 'generate_phase1.py', '--base-model', base, '--adapter', run / 'final_model',
                   '--input', sources[mode], '--output', output, '--mode', mode, '--data-version',
                   'exp4_development_draft_v1' if mode == 'fixed' else 'public_contact_masked_v1',
                   '--dtype', 'bf16', '--seed', '42', '--max-input-tokens', '1024', '--max-new-tokens', '64']
            if mode == 'fixed':
                cmd.append('--allow-unreviewed-development')
            command(cmd, dest / f'{mode}.log', 12 * 3600)
            metadata = read(output.with_suffix('.run.json'))
            if metadata['state'] != 'completed':
                raise ValueError('Generator did not complete')
            source = [json.loads(s) for s in sources[mode].read_text().splitlines()] if mode == 'fixed' else read(sources[mode])
            coverage = generation_coverage(source, output, mode)
            if metadata['generated_responses'] != coverage['responses']:
                raise ValueError('Metadata response count mismatch')
            write(dest / f'{mode}.coverage.json', coverage)
            state['completed'].append('generate:' + run.name + ':' + mode)
            status('generation_verified')
        status('4b_training_and_generation_complete_scoring_pending', current_run=None, mode=None)
    except Exception as error:
        status('stopped_on_error', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
