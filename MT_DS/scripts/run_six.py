"""Launch and supervise the six no-loss-normalization ablations."""
import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from summarize import RUNS, atomic_json, read_json, refresh


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=6,
                        help="GPUs for LA10, LA20, Pancreas10, Pancreas20, BraTS10, BraTS20")
    parser.add_argument("--memory_per_run_mib", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=Path, default=PROJECT.parent / "Dataset")
    parser.add_argument("--baseline", type=Path, default=PROJECT.parent / "MT_DS_LossNorm")
    parser.add_argument("--poll_seconds", type=float, default=30)
    args = parser.parse_args()
    if args.memory_per_run_mib < 7000 or args.poll_seconds <= 0:
        parser.error("Require >=7000 MiB per run and a positive polling interval")
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    free = {int(row.split(",")[0]): int(row.split(",")[1]) for row in raw.splitlines()}
    if args.gpus is None:
        remaining, usage = dict(free), Counter()
        args.gpus = []
        for _ in RUNS:
            candidates = [gpu for gpu in remaining if remaining[gpu] >= args.memory_per_run_mib
                          and usage[gpu] < 2]
            if not candidates:
                raise RuntimeError(f"Insufficient memory for all six experiments: {free}")
            gpu = max(candidates, key=remaining.get)
            args.gpus.append(gpu)
            remaining[gpu] -= args.memory_per_run_mib
            usage[gpu] += 1
    for gpu, count in Counter(args.gpus).items():
        if count > 2 or free.get(gpu, 0) < count * args.memory_per_run_mib:
            raise RuntimeError(f"GPU {gpu} cannot accommodate {count} runs; free MiB: {free}")
    jobs = []
    for (dataset, percent, labels), gpu in zip(RUNS, args.gpus):
        name = f"{dataset}_{labels}l"
        output = PROJECT / "Results" / f"seed_{args.seed}" / name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        parent = args.baseline / "Results" / f"seed_{args.seed}" / name
        config = read_json(parent / "config.json")
        if config.get("normalization") != "per_objective":
            raise ValueError(f"Expected the preserved normalized parent configuration: {parent}")
        command = [sys.executable, "-u", str(PROJECT / "code/train.py"),
                   "--dataset", dataset, "--labelnum", str(labels), "--seed", str(args.seed),
                   "--gpu", str(gpu), "--data_root", str(args.data_root.resolve()),
                   "--output_dir", str(output)]
        for key in ("max_iterations", "warmup_iterations", "base_lr", "ema_decay", "consistency",
                    "batch_size", "labeled_bs", "num_workers", "cpu_threads", "log_interval",
                    "val_interval", "val_start", "checkpoint_interval"):
            command.extend([f"--{key}", str(config[key])])
        command.extend(["--patch_size", *map(str, config["patch_size"])])
        jobs.append(dict(dataset=dataset, label_percent=percent, labelnum=labels, gpu=gpu,
                         output_dir=str(output), parent_config=str(parent / "config.json"), command=command))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    logs = PROJECT / "launch_logs" / stamp
    logs.mkdir(parents=True, exist_ok=False)
    manifest = dict(created_at=stamp, timezone="Asia/Shanghai", seed=args.seed,
                    supervisor_pid=os.getpid(), state="running", gpu_free_before_launch_mib=free,
                    memory_budget_per_run_mib=args.memory_per_run_mib, runs=jobs)
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    processes = []
    for job in jobs:
        logfile = logs / f"{job['dataset']}_{job['label_percent']}pct.log"
        with logfile.open("x") as handle:
            process = subprocess.Popen(job["command"], cwd=PROJECT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        job.update(pid=process.pid, launch_log=str(logfile))
        atomic_json(PROJECT / "launch_manifest.json", manifest)
        atomic_json(logs / "launch_manifest.json", manifest)
        print(f"{job['dataset']} {job['label_percent']}%: GPU={job['gpu']} PID={process.pid}", flush=True)
    while True:
        for process, job in zip(processes, jobs):
            code = process.poll()
            job["exit_code"] = code
            status_path = Path(job["output_dir"]) / "run_status.json"
            status = read_json(status_path)
            if code is not None and status.get("state") not in {"completed", "failed", "interrupted"}:
                status.update(state="failed", dataset=job["dataset"], labelnum=job["labelnum"],
                              pid=job["pid"], error=f"Training exited with code {code}; see {job['launch_log']}")
                atomic_json(status_path, status)
            job["state"] = status.get("state", "initializing")
            job["iteration"] = status.get("iteration", 0)
        rows = refresh(PROJECT, args.baseline, args.seed)
        if all(process.poll() is not None for process in processes):
            manifest["state"] = "completed" if all(row["state"] == "completed" for row in rows) else "failed"
        atomic_json(PROJECT / "launch_manifest.json", manifest)
        atomic_json(logs / "launch_manifest.json", manifest)
        if manifest["state"] != "running":
            print(f"Six-run group finished: {manifest['state']}", flush=True)
            return 0 if manifest["state"] == "completed" else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
