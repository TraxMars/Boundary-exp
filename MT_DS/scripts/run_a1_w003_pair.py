"""Launch the authorized A1 weight-only 0.03 pair without changing training code."""
import argparse
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
EXPERIMENT = PROJECT / "Results/A1_boundary_supervision_w003_v1"
RUNS = (("LA", 8), ("Pancreas", 6))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_jobs(gpus):
    jobs = []
    for (dataset, labels), gpu in zip(RUNS, gpus):
        name = f"{dataset}_{labels}l"
        source = REFERENCE / "seed_42" / name
        baseline = PROJECT / "Results/seed_42" / name
        config = read_json(source / "config.json")
        if not config or config["boundary_weight"] != 0.1 or config["seed"] != 42:
            raise RuntimeError(f"Unexpected reference configuration: {source}")
        if config["max_iterations"] != 15000 or not config["boundary_supervision"]:
            raise RuntimeError("Expected the original 15000-step A1 condition")
        hashes = read_json(source / "source_manifest.json")
        if not hashes or any(digest(PROJECT / rel) != value for rel, value in hashes.items()):
            raise RuntimeError(f"Training source differs from the 0.1 run: {dataset}")
        split = read_json(source / "data_split_manifest.json")
        if split != read_json(baseline / "data_split_manifest.json"):
            raise RuntimeError(f"Reference/Baseline split mismatch: {dataset}")
        if read_json(baseline / "run_status.json").get("state") != "completed":
            raise RuntimeError(f"Missing completed Baseline copy: {baseline}")
        for record in split["splits"].values():
            if digest(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"Dataset split file changed: {record['path']}")
        output = EXPERIMENT / "seed_42" / name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite or duplicate {output}")
        command = [sys.executable, "-u", str(PROJECT / "code/train.py"),
                   "--dataset", dataset, "--labelnum", str(labels), "--seed", "42",
                   "--gpu", str(gpu), "--data_root", config["data_root"],
                   "--output_dir", str(output), "--boundary_supervision",
                   "--boundary_weight", "0.03", "--checkpoint_interval", "100"]
        for key in ("batch_size", "labeled_bs", "max_iterations", "warmup_iterations", "base_lr",
                    "ema_decay", "consistency", "num_workers", "cpu_threads", "log_interval",
                    "val_interval", "val_start", "boundary_width", "boundary_start",
                    "boundary_ramp", "gradient_interval", "boundary_cache"):
            command += [f"--{key}", str(config[key])]
        command += ["--patch_size", *map(str, config["patch_size"])]
        jobs.append(dict(dataset=dataset, labelnum=labels, seed=42, gpu=gpu,
                         command=command, output_dir=str(output), baseline_dir=str(baseline),
                         reference_dir=str(source), source_hashes=hashes,
                         reference_split_sha256=digest(source / "data_split_manifest.json"),
                         reference_checkpoint_interval=config["checkpoint_interval"],
                         checkpoint_interval=100, initialization="random_seed_no_checkpoint",
                         algorithm_difference=dict(boundary_weight=dict(reference=0.1, current=0.03))))
    return jobs


def gpu_status(gpus):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    state = {int(parts[0]): dict(free_mib=int(parts[1]), utilization=int(parts[2]))
             for parts in (line.split(",") for line in raw.splitlines())}
    if len(set(gpus)) != 2 or any(state.get(gpu, {}).get("free_mib", 0) < 14000 for gpu in gpus):
        raise RuntimeError(f"Need two distinct GPUs with >=14000 MiB free: {state}")
    old = read_json(REFERENCE / "seed_42/Pancreas_6l/run_status.json")
    if old.get("state") != "completed" and int(old.get("gpu", -1)) in gpus:
        raise RuntimeError("Keep the currently running 0.1 Pancreas GPU separate")
    return state


def update(manifest, children, run_dir):
    for child, run in zip(children, manifest["runs"]):
        run["exit_code"] = child.poll()
        run["state"] = read_json(Path(run["output_dir"]) / "run_status.json").get("state", "initializing")
        if child.returncode is not None and (child.returncode != 0 or run["state"] != "completed"):
            run["state"] = "failed"
    if all(child.returncode is not None for child in children):
        manifest["state"] = "completed" if all(run["state"] == "completed" for run in manifest["runs"]) else "failed"
    manifest["updated_at"] = stamp()
    summary = summarize(manifest)
    summary.update(condition="A1_boundary_supervision_w003_v1", boundary_weight_maximum=0.03,
                   initialization="random_seed_no_checkpoint", pair_state=manifest["state"])
    for row, run in zip(summary["rows"], manifest["runs"]):
        row["state"] = run["state"]
    write_json(EXPERIMENT / "full_launch_manifest.json", manifest)
    write_json(run_dir / "launch_manifest.json", manifest)
    write_json(EXPERIMENT / "summary.json", summary)
    lines = ["# A1 boundary supervision, maximum weight 0.03", "", f"Updated: {stamp()}", "",
             "LA 10% and Pancreas 10%; seed 42; 15000 steps; random initialization.",
             "Only the boundary coefficient changes the algorithm; checkpoints every 100 steps.", "",
             "| Dataset | GPU | State | Step | Baseline Dice (%) | Best Dice (%) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row, run in zip(summary["rows"], manifest["runs"]):
        base = "pending" if row["baseline_dice"] is None else f"{100 * row['baseline_dice']:.4f}"
        best = "pending" if row["best_dice"] is None else f"{100 * row['best_dice']:.4f}"
        lines.append(f"| {row['dataset']} 10% | {run['gpu']} | {row['state']} | {row['iteration']} | {base} | {best} |")
    (EXPERIMENT / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--run_dir", type=Path)
    args = parser.parse_args()
    jobs = prepare_jobs(args.gpus)
    if args.check_only:
        print(json.dumps(dict(status="PASS", jobs=jobs), indent=2))
        return
    free = gpu_status(args.gpus)
    if (EXPERIMENT / "full_launch_manifest.json").exists():
        raise FileExistsError("This condition already has a launch manifest; inspect rather than relaunch")
    run_dir = args.run_dir or PROJECT / "launch_logs" / ("A1_w003_full_" + stamp().replace(":", "").replace("+", "_"))
    run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--gpus", *map(str, args.gpus),
                   "--run_dir", str(run_dir)]
        with (run_dir / "supervisor.log").open("x") as handle:
            child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=child.pid, started_at=stamp(), command=command, log_dir=str(run_dir))
        write_json(EXPERIMENT / "supervisor.json", record)
        print(json.dumps(record, indent=2), flush=True)
        return
    with (EXPERIMENT / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for run in jobs:
            run["launch_log"] = str(run_dir / f"{run['dataset']}_{run['labelnum']}l.log")
        manifest = dict(condition="A1_boundary_supervision_w003_v1", phase="full", state="running",
                        started_at=stamp(), supervisor_pid=os.getpid(), gpu_status_before_launch=free,
                        boundary_weight_maximum=0.03, checkpoint_interval=100, runs=jobs,
                        source_launcher_sha256=digest(Path(__file__)),
                        baseline_location_note="Use the preserved project Results/seed_42 copies; original path is unavailable.")
        write_json(EXPERIMENT / "full_launch_manifest.json", manifest)
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
            write_json(EXPERIMENT / "full_launch_manifest.json", manifest)
            print(f"Launched {run['dataset']} 10%: GPU {run['gpu']}, PID {child.pid}", flush=True)
        while True:
            update(manifest, children, run_dir)
            if manifest["state"] != "running":
                break
            time.sleep(30)
        if manifest["state"] != "completed":
            raise RuntimeError(f"Pair did not complete successfully; inspect {run_dir}")


if __name__ == "__main__":
    main()
