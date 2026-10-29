"""Paired smoke/full launcher for independent boundary heads, seed 42."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_a1_pair import read_json, stamp, summarize, write_json

PROJECT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT / "Results/A1_boundary_supervision_v1"
EXPERIMENT = PROJECT / "Results/A1_independent_boundary_head_w003_v1"
VERIFY = PROJECT / "verification_runs/independent_boundary_head_v1"
RUNS = (("LA", 8), ("Pancreas", 6))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(path.relative_to(PROJECT)): digest(path)
            for path in sorted((PROJECT / "code").rglob("*.py"))}


def prepare_jobs(gpus, phase, run_dir):
    jobs = []
    for (dataset, labels), gpu in zip(RUNS, gpus):
        name = f"{dataset}_{labels}l"
        reference = REFERENCE / "seed_42" / name
        baseline = PROJECT / "Results/seed_42" / name
        config = read_json(reference / "config.json")
        original_hashes = read_json(reference / "source_manifest.json")
        if not original_hashes or any(digest(PROJECT / key) != value for key, value in original_hashes.items()):
            raise RuntimeError("Original training sources changed")
        if config.get("max_iterations") != 15000 or config.get("seed") != 42:
            raise RuntimeError("Unexpected reference")
        split = read_json(reference / "data_split_manifest.json")
        if split != read_json(baseline / "data_split_manifest.json"):
            raise RuntimeError(f"Reference/Baseline split mismatch: {dataset}")
        for record in split["splits"].values():
            if digest(Path(record["path"])) != record["sha256"]:
                raise RuntimeError("Dataset case list changed")
        output = (EXPERIMENT / "seed_42" / name if phase == "full" else run_dir / "runs" / name)
        if output.exists():
            raise FileExistsError(f"Refusing overwrite/duplicate launch: {output}")
        command = [sys.executable, "-u", str(PROJECT / "code/train_boundary_head.py"),
                   "--dataset", dataset, "--labelnum", str(labels), "--seed", "42",
                   "--gpu", str(gpu), "--data_root", config["data_root"], "--output_dir", str(output),
                   "--boundary_supervision", "--boundary_weight", "0.03", "--checkpoint_interval", "100"]
        for key in ("batch_size", "labeled_bs", "base_lr", "ema_decay", "consistency", "num_workers",
                    "cpu_threads", "boundary_width", "boundary_cache"):
            command += [f"--{key}", str(config[key])]
        command += ["--patch_size", *map(str, config["patch_size"])]
        schedule = (dict(max_iterations=8, warmup_iterations=2, log_interval=1, val_interval=0,
                         val_start=800, boundary_start=2, boundary_ramp=2, gradient_interval=2,
                         checkpoint_interval=4) if phase == "smoke" else
                    dict(max_iterations=15000, warmup_iterations=2000, log_interval=10, val_interval=200,
                         val_start=800, boundary_start=2000, boundary_ramp=2000, gradient_interval=200))
        for key, value in schedule.items():
            command += [f"--{key}", str(value)]
        jobs.append(dict(dataset=dataset, labelnum=labels, seed=42, gpu=gpu, command=command,
                         output_dir=str(output), reference_dir=str(reference), baseline_dir=str(baseline),
                         launch_log=str(run_dir / f"{name}.log"), initialization="random_seed_no_checkpoint"))
    return jobs


def gpu_status(gpus):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    state = {int(parts[0]): dict(free_mib=int(parts[1]), utilization=int(parts[2]))
             for parts in (line.split(",") for line in raw.splitlines())}
    if len(set(gpus)) != 2 or any(state.get(gpu, {}).get("free_mib", 0) < 14000 for gpu in gpus):
        raise RuntimeError(f"Need two distinct GPUs with >=14000 MiB free: {state}")
    for name in ("LA_8l", "Pancreas_6l"):
        status = read_json(PROJECT / "Results/A1_boundary_supervision_w003_v1/seed_42" / name / "run_status.json")
        if status.get("state") in ("training", "validating") and int(status.get("gpu", -1)) in gpus:
            raise RuntimeError("Keep new jobs separate from the running A1 weight-only pair")
    return state


def verify_gate(path, hashes):
    record = read_json(path)
    if record.get("status") != "PASS" or record.get("source_hashes") != hashes:
        raise RuntimeError(f"Missing/stale verification: {path}")


def update(manifest, children, run_dir):
    destination = Path(manifest["result_dir"])
    for child, run in zip(children, manifest["runs"]):
        run["exit_code"] = child.poll()
        run["state"] = read_json(Path(run["output_dir"]) / "run_status.json").get("state", "initializing")
        if child.returncode is not None and (child.returncode != 0 or run["state"] != "completed"):
            run["state"] = "failed"
    if all(child.returncode is not None for child in children):
        manifest["state"] = "completed" if all(run["state"] == "completed" for run in manifest["runs"]) else "failed"
    manifest["updated_at"] = stamp()
    summary = summarize(manifest)
    summary.update(condition="A1_independent_boundary_head_w003_v1", pair_state=manifest["state"],
                   boundary_weight_maximum=.03, boundary_task="binary_boundary_nonboundary")
    for row, run in zip(summary["rows"], manifest["runs"]):
        row["state"] = run["state"]
    write_json(destination / "full_launch_manifest.json", manifest)
    write_json(run_dir / "launch_manifest.json", manifest)
    write_json(destination / "summary.json", summary)
    lines = ["# Independent boundary heads, half/full scales", "", f"Updated: {stamp()}", "",
             "Boundary BCE+Dice maximum weight 0.03; seed 42; original segmentation inference.", "",
             "| Dataset | GPU | State | Step | Baseline Dice (%) | Best Dice (%) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row, run in zip(summary["rows"], manifest["runs"]):
        baseline = "pending" if row["baseline_dice"] is None else f"{100*row['baseline_dice']:.4f}"
        best = "pending" if row["best_dice"] is None else f"{100*row['best_dice']:.4f}"
        lines.append(f"| {row['dataset']} 10% | {run['gpu']} | {row['state']} | {row['iteration']} | {baseline} | {best} |")
    (destination / "summary.md").write_text("\n".join(lines) + "\n")


def verify_smoke(manifest):
    results = []
    for run in manifest["runs"]:
        output = Path(run["output_dir"])
        status = read_json(output / "run_status.json")
        if run["exit_code"] != 0 or status.get("state") != "completed" or status.get("iteration") != 8:
            raise RuntimeError(f"Incomplete paired smoke: {run['dataset']}")
        rows = list(csv.DictReader((output / "training_metrics.csv").open()))
        probes = [json.loads(line) for line in (output / "boundary_gradient_metrics.jsonl").read_text().splitlines()]
        if len(rows) != 8 or not any(float(row["boundary_weight"]) > 0 for row in rows):
            raise RuntimeError("Smoke did not exercise active boundary learning")
        if not any(row["boundary_weight"] > 0 and row["boundary_norm"] > 0 for row in probes):
            raise RuntimeError("Boundary loss did not reach the backbone")
        if read_json(output / "data_split_manifest.json") != read_json(Path(run["reference_dir"]) / "data_split_manifest.json"):
            raise RuntimeError("Smoke split changed")
        results.append(dict(dataset=run["dataset"], output_dir=str(output),
                            loss_total=float(rows[-1]["loss_total"]),
                            peak_memory_mib=max(float(row["peak_memory_mib"]) for row in rows)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--run_dir", type=Path)
    args = parser.parse_args()
    hashes = source_hashes()
    verify_gate(VERIFY / "unit_results.json", hashes)
    if args.phase == "full":
        verify_gate(VERIFY / "smoke_results.json", hashes)
    suffix = stamp().replace(":", "").replace("+", "_")
    run_dir = args.run_dir or ((VERIFY / ("smoke_" + suffix)) if args.phase == "smoke" else
                               PROJECT / "launch_logs" / ("independent_boundary_head_" + suffix))
    jobs = prepare_jobs(args.gpus, args.phase, run_dir)
    if args.check_only:
        print(json.dumps(dict(status="PASS", jobs=jobs), indent=2))
        return
    free = gpu_status(args.gpus)
    destination = EXPERIMENT if args.phase == "full" else run_dir
    if (destination / "full_launch_manifest.json").exists():
        raise FileExistsError("Launch already recorded; do not duplicate")
    run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
    destination.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--gpus", *map(str, args.gpus),
                   "--phase", args.phase, "--run_dir", str(run_dir)]
        with (run_dir / "supervisor.log").open("x") as handle:
            child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=child.pid, started_at=stamp(), command=command, log_dir=str(run_dir))
        write_json(destination / "supervisor.json", record)
        print(json.dumps(record, indent=2), flush=True)
        return
    with (destination / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(condition="A1_independent_boundary_head_w003_v1", phase=args.phase,
                        state="running", started_at=stamp(), supervisor_pid=os.getpid(),
                        result_dir=str(destination), gpu_status_before_launch=free, runs=jobs,
                        source_hashes=hashes, source_launcher_sha256=digest(Path(__file__)),
                        boundary_weight_maximum=.03, checkpoint_interval=100,
                        algorithm="original_MT_DS_plus_independent_half_full_boundary_heads",
                        inference="original_segmentation_head_only")
        write_json(destination / "full_launch_manifest.json", manifest)
        environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                           PYTHONUNBUFFERED="1")
        children = []
        for run in jobs:
            with Path(run["launch_log"]).open("x") as handle:
                child = subprocess.Popen(run["command"], cwd=PROJECT, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=handle,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            run.update(pid=child.pid, state="initializing", exit_code=None)
            write_json(destination / "full_launch_manifest.json", manifest)
            print(f"Launched {run['dataset']} 10% ({args.phase}): GPU {run['gpu']}, PID {child.pid}", flush=True)
        while True:
            update(manifest, children, run_dir)
            if manifest["state"] != "running":
                break
            time.sleep(2 if args.phase == "smoke" else 30)
        if manifest["state"] != "completed":
            raise RuntimeError(f"Paired {args.phase} failed; inspect {run_dir}")
        if args.phase == "smoke":
            runs = verify_smoke(manifest)
            write_json(VERIFY / "smoke_results.json", dict(status="PASS", checked_at=stamp(),
                                                          source_hashes=hashes, runs=runs))
            print(json.dumps(dict(status="PASS", runs=runs), indent=2), flush=True)


if __name__ == "__main__":
    main()
