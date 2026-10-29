"""Resume an interrupted A1 member while preserving the completed partner."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from run_a1_pair import read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = PROJECT / "Results/A1_boundary_supervision_v1"


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def matching_process(pid, token):
    try:
        return token in (Path("/proc") / str(pid) / "cmdline").read_bytes().decode(errors="replace")
    except (OSError, TypeError):
        return False


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def trim_metrics(path, checkpoint_step):
    if not path.exists():
        return 0
    if path.suffix == ".csv":
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            original = list(reader)
        keep = [row for row in original if int(row["iteration"]) <= checkpoint_step]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(keep)
        text = output.getvalue()
    else:
        original = path.read_text().splitlines()
        keep = [line for line in original if int(json.loads(line)["iteration"]) <= checkpoint_step]
        text = "\n".join(keep) + ("\n" if keep else "")
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    temporary.write_text(text)
    temporary.replace(path)
    return len(original) - len(keep)


def update_summary(manifest, active_run, child):
    rows = []
    for run in manifest["runs"]:
        output = Path(run["output_dir"])
        status = read_json(output / "run_status.json")
        best = read_json(output / "best_metrics.json")
        final = read_json(output / "test_metrics.json")
        baseline = read_json(Path(run["baseline_dir"]) / "test_metrics.json")
        state = status.get("state", "unknown")
        if run is active_run:
            run["exit_code"] = child.poll()
            run["state"] = state
            if child.poll() is not None and state != "completed":
                state = "failed" if child.returncode else "incomplete"
        elif (state != "completed" and status.get("iteration", 0) >= status.get("max_iterations", 15000)
              and not matching_process(status.get("pid"), "/code/train.py")):
            state = "training_finished_finalization_pending"
        rows.append(dict(dataset=run["dataset"], labelnum=run["labelnum"], state=state,
                         iteration=status.get("iteration", 0), pid=run.get("pid"),
                         baseline_dice=baseline.get("dice"), best_dice=best.get("dice"),
                         test_dice=final.get("dice"), test_hd95=final.get("hd95"),
                         test_asd=final.get("asd"), exit_code=run.get("exit_code"),
                         output_dir=str(output), launch_log=run["launch_log"]))
    summary = dict(updated_at=timestamp(), phase="full", protocol="legacy_test_selection_exploratory", rows=rows)
    write_json(EXPERIMENT / "summary.json", summary)
    lines = ["# A1 boundary supervision", "", f"Updated: {timestamp()}", "",
             "Protocol: legacy test-selection, exploratory; distances in voxels.", "",
             "| Dataset | State | Iteration | Baseline Dice (%) | Best Dice (%) |",
             "| --- | --- | --- | --- | --- |"]
    for row in rows:
        best = f"{100 * row['best_dice']:.4f}" if row["best_dice"] is not None else "pending"
        lines.append(f"| {row['dataset']} 10% | {row['state']} | {row['iteration']} | "
                     f"{100 * row['baseline_dice']:.4f} | {best} |")
    (EXPERIMENT / "summary.md").write_text("\n".join(lines) + "\n")
    manifest["updated_at"] = timestamp()
    manifest["state"] = ("running" if child.poll() is None else
                         "completed" if all(row["state"] == "completed" for row in rows) else
                         "partial_completion" if child.returncode == 0 else "failed")
    write_json(EXPERIMENT / "full_launch_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Pancreas", "LA"), required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--run_dir", type=Path)
    args = parser.parse_args()
    if args.checkpoint_interval <= 0:
        parser.error("checkpoint_interval must be positive")
    labels = 6 if args.dataset == "Pancreas" else 8
    output = EXPERIMENT / "seed_42" / f"{args.dataset}_{labels}l"
    config = read_json(output / "config.json")
    status = read_json(output / "run_status.json")
    if not config or status.get("state") == "completed":
        raise RuntimeError("No interrupted run to resume")
    if matching_process(status.get("pid"), "/code/train.py"):
        raise RuntimeError("The old training process still exists")
    supervisor = read_json(EXPERIMENT / "supervisor.json")
    if matching_process(supervisor.get("pid"), "scripts/run_a1_pair.py") or matching_process(
            supervisor.get("pid"), "scripts/resume_a1.py"):
        # The detached supervisor checks its own PID here; only other processes block.
        if supervisor.get("pid") != os.getpid():
            raise RuntimeError("An A1 supervisor is still active")
    for rel, saved in read_json(output / "source_manifest.json").items():
        if digest(PROJECT / rel) != saved:
            raise RuntimeError(f"Training source differs from the original run: {rel}")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    gpus = [tuple(int(x.strip()) for x in line.split(",")) for line in raw.splitlines()]
    eligible = [row for row in gpus if row[1] >= 12000 and (args.gpu is None or row[0] == args.gpu)]
    if not eligible:
        raise RuntimeError("No GPU with >= 12000 MiB free")
    gpu = min(eligible, key=lambda row: (row[2], -row[1]))[0]
    run_dir = args.run_dir or PROJECT / "launch_logs" / (
        f"A1_resume_{args.dataset}_" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--dataset", args.dataset,
                   "--gpu", str(gpu), "--checkpoint_interval", str(args.checkpoint_interval),
                   "--run_dir", str(run_dir)]
        with (run_dir / "supervisor.log").open("x") as handle:
            child = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(run_dir / "supervisor.json", dict(pid=child.pid, command=command, started_at=timestamp()))
        print(json.dumps(dict(supervisor_pid=child.pid, log_dir=str(run_dir), gpu=gpu), indent=2))
        return

    import torch
    checkpoint = output / "last_checkpoint.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(saved["iteration"])
    if not 0 < step < int(config["max_iterations"]):
        raise RuntimeError(f"Checkpoint step {step} is not an interrupted training interval")
    for key in ("student", "teacher", "optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"):
        if key not in saved:
            raise RuntimeError(f"Checkpoint missing {key}")
    best = read_json(output / "best_metrics.json")
    if best and (best.get("iteration", step + 1) > step or abs(best["dice"] - saved["best_dice"]) > 1e-10):
        raise RuntimeError("Best model metadata is newer than checkpoint; reconcile before resuming")
    archive = output / "resume_history" / run_dir.name
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("run_status.json", "config.json", "training_metrics.csv", "validation_metrics.jsonl",
                 "boundary_gradient_metrics.jsonl", "latest_validation.json", "best_metrics.json", "log.txt"):
        if (output / name).exists():
            shutil.copy2(output / name, archive / name)
    shutil.copy2(checkpoint, archive / checkpoint.name)
    shutil.copy2(EXPERIMENT / "full_launch_manifest.json", archive / "full_launch_manifest.json")
    if (EXPERIMENT / "supervisor.json").exists():
        shutil.copy2(EXPERIMENT / "supervisor.json", archive / "supervisor.json")
    removed = {name: trim_metrics(output / name, step) for name in (
        "training_metrics.csv", "validation_metrics.jsonl", "boundary_gradient_metrics.jsonl")}
    validation_path = output / "validation_metrics.jsonl"
    if validation_path.exists() and validation_path.read_text().strip():
        write_json(output / "latest_validation.json", json.loads(validation_path.read_text().splitlines()[-1]))
    if (output / "tensorboard").exists():
        (output / "tensorboard").rename(archive / "tensorboard_before_resume")

    manifest = read_json(EXPERIMENT / "full_launch_manifest.json")
    active = next(run for run in manifest["runs"] if run["dataset"] == args.dataset)
    command = list(active["command"])
    def set_argument(name, value):
        if name in command:
            command[command.index(name) + 1] = str(value)
        else:
            command.extend((name, str(value)))
    set_argument("--gpu", gpu)
    set_argument("--resume", checkpoint)
    set_argument("--checkpoint_interval", args.checkpoint_interval)
    resumed = dict(dataset=args.dataset, resumed_at=timestamp(), checkpoint=str(checkpoint),
                   checkpoint_sha256=digest(checkpoint), checkpoint_iteration=step,
                   previous_logged_iteration=status.get("iteration"),
                   rollback_updates=status.get("iteration", step) - step,
                   removed_live_rows=removed, archive_dir=str(archive),
                   checkpoint_interval_before=config["checkpoint_interval"],
                   checkpoint_interval_now=args.checkpoint_interval,
                   note="Same optimizer/EMA/objectives/schedule; DataLoader batches are not bitwise restored.",
                   command=command, gpu=gpu, supervisor_pid=os.getpid())
    write_json(archive / "resume_record.json", resumed)
    with (output / "log.txt").open("a") as handle:
        handle.write(f"\nRESUME {timestamp()}: restoring step {step}; previous logged step "
                     f"{status.get('iteration')}; archival data in {archive}; checkpoint interval "
                     f"{args.checkpoint_interval}.\n")
    launch_log = run_dir / f"{args.dataset}_{labels}l.log"
    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                       PYTHONUNBUFFERED="1")
    with launch_log.open("x") as handle:
        child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    resumed["pid"] = child.pid
    write_json(archive / "resume_record.json", resumed)
    write_json(run_dir / "resume_record.json", resumed)
    active.update(command=command, pid=child.pid, gpu=gpu, launch_log=str(launch_log),
                  exit_code=None, state="resuming", resume_record=str(archive / "resume_record.json"))
    manifest["supervisor_pid"] = os.getpid()
    manifest.setdefault("resume_history", []).append(resumed)
    write_json(EXPERIMENT / "supervisor.json", dict(pid=os.getpid(), phase="resume", dataset=args.dataset,
               started_at=timestamp(), log_dir=str(run_dir), command=[sys.executable, *sys.argv]))
    print(json.dumps(resumed, indent=2), flush=True)
    while True:
        update_summary(manifest, active, child)
        write_json(run_dir / "launch_manifest.json", manifest)
        if child.poll() is not None:
            break
        time.sleep(30)
    if child.returncode:
        raise RuntimeError(f"Resumed training failed; inspect {launch_log}")


if __name__ == "__main__":
    main()
