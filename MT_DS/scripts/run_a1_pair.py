"""Run the authorized A1 pair, with numerical and paired-smoke verification."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[1]
VERIFY = PROJECT / "verification_runs/A1_boundary_v1"
EXPERIMENT = PROJECT / "Results/A1_boundary_supervision_v1"
RUNS = (("LA", 8), ("Pancreas", 6))


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("code", "scripts", "tests")
            for p in sorted((PROJECT / directory).rglob("*.py"))}


def validate_gate(path, hashes):
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("source_hashes") != hashes:
        raise RuntimeError(f"Missing/stale verification: {path}")


def summarize(manifest):
    rows = []
    for run in manifest["runs"]:
        output = Path(run["output_dir"])
        state = read_json(output / "run_status.json")
        best = read_json(output / "best_metrics.json")
        final = read_json(output / "test_metrics.json")
        baseline = read_json(Path(run["baseline_dir"]) / "test_metrics.json")
        row = dict(dataset=run["dataset"], labelnum=run["labelnum"],
                   state=state.get("state", run.get("state", "initializing")),
                   iteration=state.get("iteration", 0), pid=run.get("pid"),
                   baseline_dice=baseline.get("dice"), best_dice=best.get("dice"),
                   test_dice=final.get("dice"), test_hd95=final.get("hd95"),
                   test_asd=final.get("asd"), exit_code=run.get("exit_code"),
                   output_dir=str(output), launch_log=run["launch_log"])
        if final and baseline:
            row["dice_delta_pp"] = 100 * (final["dice"] - baseline["dice"])
        rows.append(row)
    return dict(updated_at=stamp(), phase=manifest["phase"],
                protocol="legacy_test_selection_exploratory", rows=rows)


def validate_smoke(manifest):
    results = []
    for run in manifest["runs"]:
        output = Path(run["output_dir"])
        state = read_json(output / "run_status.json")
        if run["exit_code"] != 0 or state.get("state") != "completed" or state.get("iteration") != 8:
            raise RuntimeError(f"Smoke run did not complete: {run['dataset']}")
        with (output / "training_metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 8 or not any(float(r["boundary_weight"]) > 0 for r in rows):
            raise RuntimeError("Smoke must exercise the nonzero boundary term")
        if not any(float(r["loss_boundary"]) > 0 for r in rows):
            raise RuntimeError("Smoke did not observe any valid boundary")
        probes = [json.loads(line) for line in (output / "boundary_gradient_metrics.jsonl").read_text().splitlines()]
        if not any(p["boundary_weight"] > 0 and p["boundary_norm"] > 0 for p in probes):
            raise RuntimeError("No active finite boundary gradient in smoke")
        if not (output / "last_checkpoint.pth").is_file():
            raise RuntimeError("Smoke did not save a checkpoint")
        current_split = read_json(output / "data_split_manifest.json")
        baseline_split = read_json(Path(run["baseline_dir"]) / "data_split_manifest.json")
        if current_split != baseline_split:
            raise RuntimeError("Smoke changed the Baseline case split")
        results.append(dict(dataset=run["dataset"], output_dir=str(output),
                            final_loss=float(rows[-1]["loss_total"]),
                            peak_memory_mib=max(float(r["peak_memory_mib"]) for r in rows),
                            active_boundary_probes=sum(p["boundary_weight"] > 0 and p["boundary_norm"] > 0 for p in probes)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--gpus", type=int, nargs=2)
    parser.add_argument("--minimum_free_mib", type=int, default=14000)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--run_dir", type=Path)
    args = parser.parse_args()
    hashes = source_hashes()
    validate_gate(VERIFY / "unit_results.json", hashes)
    if args.phase == "full":
        validate_gate(VERIFY / "smoke_results.json", hashes)
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                                   "--format=csv,noheader,nounits"], text=True)
    free = {int(line.split(",")[0]): int(line.split(",")[1]) for line in raw.splitlines()}
    if args.gpus is None:
        args.gpus = sorted((gpu for gpu in free if free[gpu] >= args.minimum_free_mib),
                           key=lambda gpu: free[gpu], reverse=True)[:2]
    if len(args.gpus) != 2 or len(set(args.gpus)) != 2:
        raise RuntimeError("Require two distinct GPUs with sufficient free memory")
    if any(free.get(gpu, 0) < args.minimum_free_mib for gpu in args.gpus):
        raise RuntimeError(f"Insufficient free memory on GPUs {args.gpus}: {free}")
    run_dir = args.run_dir or PROJECT / "launch_logs" / (
        "A1_" + args.phase + "_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", args.phase,
                   "--gpus", *map(str, args.gpus), "--minimum_free_mib", str(args.minimum_free_mib),
                   "--run_dir", str(run_dir)]
        with (run_dir / "supervisor.log").open("x") as handle:
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=process.pid, phase=args.phase, started_at=stamp(),
                      log_dir=str(run_dir), command=command)
        write_json((EXPERIMENT if args.phase == "full" else VERIFY) / "supervisor.json", record)
        print(json.dumps(record, indent=2))
        return
    jobs = []
    for (dataset, labels), gpu in zip(RUNS, args.gpus):
        name = f"{dataset}_{labels}l"
        baseline_dir = WORKSPACE / "MT_DS/Results/seed_42" / name
        baseline = read_json(baseline_dir / "config.json")
        if not baseline or read_json(baseline_dir / "run_status.json").get("state") != "completed":
            raise RuntimeError(f"Missing completed Baseline: {baseline_dir}")
        output = (EXPERIMENT / "seed_42" / name if args.phase == "full" else
                  VERIFY / run_dir.name / name)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        command = [sys.executable, "-u", str(PROJECT / "code/train.py"), "--dataset", dataset,
                   "--labelnum", str(labels), "--seed", "42", "--gpu", str(gpu),
                   "--data_root", str(WORKSPACE / "Dataset"), "--output_dir", str(output),
                   "--boundary_supervision", "--boundary_width", "2", "--boundary_weight", "0.1"]
        for key in ("batch_size", "labeled_bs", "base_lr", "ema_decay", "consistency",
                    "num_workers", "cpu_threads"):
            command += [f"--{key}", str(baseline[key])]
        command += ["--patch_size", *map(str, baseline["patch_size"])]
        if args.phase == "smoke":
            command += ["--max_iterations", "8", "--warmup_iterations", "2", "--boundary_start", "2",
                        "--boundary_ramp", "2", "--gradient_interval", "2", "--val_interval", "0",
                        "--checkpoint_interval", "8", "--log_interval", "1"]
        else:
            for key in ("max_iterations", "warmup_iterations", "val_interval", "val_start",
                        "checkpoint_interval", "log_interval"):
                command += [f"--{key}", str(baseline[key])]
            command += ["--boundary_start", "2000", "--boundary_ramp", "2000", "--gradient_interval", "200"]
        jobs.append(dict(dataset=dataset, labelnum=labels, seed=42, gpu=gpu, command=command,
                         output_dir=str(output), baseline_dir=str(baseline_dir),
                         launch_log=str(run_dir / f"{name}.log")))
    manifest = dict(phase=args.phase, started_at=stamp(), supervisor_pid=os.getpid(),
                    state="running", gpu_free_before_launch_mib=free, source_hashes=hashes, runs=jobs)
    destination = EXPERIMENT if args.phase == "full" else VERIFY
    subprocesses = []
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1")
    for run in jobs:
        with Path(run["launch_log"]).open("x") as handle:
            child = subprocess.Popen(run["command"], cwd=PROJECT, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        subprocesses.append(child)
        run["pid"] = child.pid
        print(f"Launched {run['dataset']} 10%: GPU {run['gpu']}, PID {child.pid}", flush=True)
    while True:
        for child, run in zip(subprocesses, jobs):
            run["exit_code"] = child.poll()
            if run["exit_code"] is not None:
                run["state"] = "completed" if run["exit_code"] == 0 else "failed"
        summary = summarize(manifest)
        if all(child.poll() is not None for child in subprocesses):
            manifest["state"] = "completed" if all(
                row["state"] == "completed" and row["exit_code"] == 0 for row in summary["rows"]) else "failed"
        manifest["updated_at"] = stamp()
        write_json(run_dir / "launch_manifest.json", manifest)
        write_json(destination / f"{args.phase}_launch_manifest.json", manifest)
        write_json(destination / "summary.json", summary)
        lines = ["# A1 boundary supervision", "", f"Updated: {stamp()}", "",
                 "Protocol: legacy test-selection, exploratory; distances in voxels.", "",
                 "| Dataset | State | Iteration | Baseline Dice (%) | Best Dice (%) |",
                 "| --- | --- | --- | --- | --- |"]
        for row in summary["rows"]:
            best = f"{100 * row['best_dice']:.4f}" if row["best_dice"] is not None else "pending"
            lines.append(f"| {row['dataset']} 10% | {row['state']} | {row['iteration']} | "
                         f"{100 * row['baseline_dice']:.4f} | {best} |")
        (destination / "summary.md").write_text("\n".join(lines) + "\n")
        if manifest["state"] != "running":
            break
        time.sleep(5 if args.phase == "smoke" else 30)
    if manifest["state"] != "completed":
        raise RuntimeError(f"Paired {args.phase} failed; inspect {run_dir}")
    if args.phase == "smoke":
        results = validate_smoke(manifest)
        write_json(VERIFY / "smoke_results.json", dict(status="PASS", source_hashes=hashes,
                                                       checked_at=stamp(), runs=results))
        print(json.dumps(dict(status="PASS", runs=results), indent=2), flush=True)


if __name__ == "__main__":
    main()
