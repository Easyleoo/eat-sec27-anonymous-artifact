#!/usr/bin/env python3
"""Bounded seed-42 14B replication, with independent download and shutdown guards."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_phase1_4b_queue import generation_coverage

GIB = 1024 ** 3


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def remaining_seconds(deadline):
    return (deadline - datetime.now(timezone.utc)).total_seconds()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--deadline", required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--shutdown-on-stop", action="store_true")
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.utcoffset() is None or remaining_seconds(deadline) <= 0:
        raise ValueError("A future timezone-aware deadline is required")
    root = args.root.resolve()
    runtime = Path(__file__).resolve().parent
    py = root / "envs/train4b/bin/python"
    inputs = root / "inputs/exp1_matched_public_v1"
    base = root / "models/Qwen3-14B"
    record = root / "records/phase1_14b_state.json"
    plan = {"model": "Qwen/Qwen3-14B", "arms": ["clean", "are", "rand_family", "raw"],
        "seed": 42, "epochs": 4, "batch_size": 4, "grad_accum": 4,
        "dtype": "bf16", "gradient_checkpointing": True, "save_best_checkpoint": False,
        "checkpoint_for_analysis": "final_model", "deadline": deadline.isoformat(),
        "shutdown_on_stop": args.shutdown_on_stop, "judge_called": False,
        "test_output_policy": "No test-dependent checkpoint, configuration or training selection"}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    import fcntl
    lock = (root / "records/phase1_14b.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if record.exists():
        raise FileExistsError("Inspect the prior queue; do not silently restart")
    state = {**plan, "state": "starting", "completed": []}
    write(root / "records/phase1_14b_plan.json", plan)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
           "MKL_NUM_THREADS": "8", "PYTHONUNBUFFERED": "1"}

    def status(stage, **extra):
        state.update(state=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write(record, state)

    def command(cmd, log, timeout):
        with log.open("x") as out:
            process = subprocess.Popen([str(x) for x in cmd], stdout=out, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            status(state["state"], child_pid=process.pid)
            started = time.monotonic()
            while process.poll() is None:
                if remaining_seconds(deadline) < 180 or time.monotonic() - started > timeout:
                    # Training handles SIGTERM at an optimizer boundary and saves a partial adapter.
                    process.terminate()
                    try:
                        process.wait(timeout=150)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise TimeoutError("Job stopped at the bounded budget; partial output is not complete")
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f"Child failed with exit {process.returncode}; see {log.name}")
            status(state["state"], child_pid=None)

    def verify(run, steps):
        import torch
        from safetensors import safe_open
        torch.set_num_threads(4)
        summary = read(run / "training_summary.json")
        if summary["state"] != "completed" or summary["global_step"] != steps:
            raise ValueError("Training steps are incomplete")
        count = 0
        with safe_open(str(run / "final_model/adapter_model.safetensors"), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                tensor = weights.get_tensor(name)
                if not torch.isfinite(tensor).all().item():
                    raise ValueError("Nonfinite saved parameters")
                count += int("lora_B" in name and torch.count_nonzero(tensor).item() > 0)
        if not count:
            raise ValueError("No trained adapter parameters")
        output = run / "development_reload_check.jsonl"
        command([py, runtime / "generate_phase1.py", "--base-model", base,
                 "--adapter", run / "final_model", "--input", sources["fixed"], "--output", output,
                 "--mode", "fixed", "--data-version", "exp4_development_draft_v1",
                 "--allow-unreviewed-development", "--dtype", "bf16", "--limit", "4",
                 "--max-new-tokens", "64"], run / "reload_check.log", 900)
        replies = [json.loads(x) for x in output.read_text().splitlines()]
        if len(replies) != 4 or not all(r["response_text"].strip() for r in replies):
            raise ValueError("Reload check failed")
        write(run / "verification.json", {"state": "passed", "finite_saved_weights": True,
              "nonzero_lora_B_tensors": count, "adapter_reloaded": True, "formal_evaluation": "pending"})

    def train(run, train_file, valid_file, epochs):
        run.mkdir(exist_ok=False)
        shutil.copyfile(runtime / "train_lora_rebuttal.py", run / "training_script.py")
        command([py, "-u", run / "training_script.py", "--train_file", train_file,
                 "--valid_file", valid_file, "--model_name", base, "--output_dir", run,
                 "--offline", "--use_lora", "--bf16", "--gradient_checkpointing",
                 "--skip_best_checkpoint", "--epochs", str(epochs), "--batch_size", "4",
                 "--grad_accum", "4", "--block_size", "1024", "--seed", "42", "--lr", "5e-5",
                 "--logging_steps", "20", "--eval_steps", "200", "--lora_r", "16",
                 "--lora_alpha", "32", "--lora_dropout", "0.05"], run / "training.log", 3 * 3600)

    try:
        frozen = runtime / "frozen_generation_inputs"
        frozen.mkdir(exist_ok=False)
        sources = {"fixed": root / "code/data/exp4/exp4_development_draft.jsonl",
                   "dynamic": root / "code/data/released/sad_test_1000.json",
                   "neutral": root / "code/data/released/neutral_test_1000.json"}
        for mode, source in list(sources.items()):
            shutil.copyfile(source, frozen / source.name)
            sources[mode] = frozen / source.name
        for arm in plan["arms"]:
            if len(read(inputs / f"train_{arm}.json")) != 10000:
                raise ValueError("Training data count changed")
        status("waiting_for_download")
        download_deadline = time.monotonic() + 45 * 60
        while True:
            download = read(root / "records/qwen14b_download_status.json")
            if download["state"] == "ready":
                break
            if download["state"] == "failed":
                raise RuntimeError("Download failed")
            if time.monotonic() > download_deadline or remaining_seconds(deadline) < 2 * 3600:
                raise TimeoutError("Stop idle GPU billing: model preparation exceeded its time budget")
            time.sleep(15)
        smoke_inputs = runtime / "smoke_inputs"
        smoke_inputs.mkdir(exist_ok=False)
        # Stress the loader with the longest clean training conversations; never use test records.
        rows = sorted(read(inputs / "train_clean.json"),
                      key=lambda r: len(json.dumps(r, ensure_ascii=False)), reverse=True)[:32]
        write(smoke_inputs / "train.json", rows)
        write(smoke_inputs / "validation.json", read(inputs / "validation_clean.json")[:8])
        smoke = root / "runs/exp1/14b_smoke_bf16_v1"
        status("smoke_training", current_run=smoke.name)
        train(smoke, smoke_inputs / "train.json", smoke_inputs / "validation.json", 1)
        verify(smoke, 2)
        state["completed"].append("smoke")
        for arm in plan["arms"]:
            if remaining_seconds(deadline) < 2.5 * 3600:
                status("stopped_before_new_run_budget", pending_arm=arm)
                break
            if shutil.disk_usage(root).free < 8 * GIB:
                raise RuntimeError("Less than 8 GiB free: retain all existing results and stop")
            run = root / f"runs/exp1/14b_{arm}_seed42_public_bf16_v1"
            status("training", current_run=run.name, mode=None)
            train(run, inputs / f"train_{arm}.json", inputs / "validation_clean.json", 4)
            status("verifying", current_run=run.name)
            verify(run, 2500)
            state["completed"].append("train:" + run.name)
            modes = (["fixed"] if arm in {"clean", "are"} else []) + ["dynamic", "neutral"]
            dest = run / "generation_v1"
            dest.mkdir()
            for mode in modes:
                status("generating", current_run=run.name, mode=mode)
                output = dest / f"{mode}.jsonl"
                cmd = [py, runtime / "generate_phase1.py", "--base-model", base,
                       "--adapter", run / "final_model", "--input", sources[mode], "--output", output,
                       "--mode", mode, "--data-version", "exp4_development_draft_v1" if mode == "fixed" else "public_contact_masked_v1",
                       "--dtype", "bf16", "--seed", "42", "--max-input-tokens", "1024", "--max-new-tokens", "64"]
                if mode == "fixed":
                    cmd.append("--allow-unreviewed-development")
                command(cmd, dest / f"{mode}.log", 2 * 3600)
                source = [json.loads(x) for x in sources[mode].read_text().splitlines()] if mode == "fixed" else read(sources[mode])
                coverage = generation_coverage(source, output, mode)
                metadata = read(output.with_suffix(".run.json"))
                if metadata["state"] != "completed" or metadata["generated_responses"] != coverage["responses"]:
                    raise ValueError("Generation metadata/coverage mismatch")
                write(dest / f"{mode}.coverage.json", coverage)
                state["completed"].append("generate:" + run.name + ":" + mode)
                status("generation_verified")
        else:
            status("14b_training_and_generation_complete_scoring_pending", current_run=None, mode=None)
    except Exception as error:
        status("stopped_on_error", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        os.sync()
        if args.shutdown_on_stop:
            state["shutdown_requested_at"] = datetime.now(timezone.utc).isoformat()
            write(record, state)
            os.sync()
            subprocess.run(["/bin/bash", "/usr/bin/shutdown"], timeout=45, check=False)


if __name__ == "__main__":
    main()
