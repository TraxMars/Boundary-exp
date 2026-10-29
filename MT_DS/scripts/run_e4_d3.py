"""Launch the two authorized boundary conditions on both 10% datasets."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from run_a1_pair import read_json, stamp, write_json

PROJECT = Path(__file__).resolve().parents[1]
VERIFY = PROJECT / "verification_runs/boundary_probe_v1"
COMPARISON = PROJECT / "Results/E4_D3_comparison_v1"
CONDITIONS = dict(e4="E4_legacy_boundary_w003_v1", d3="D3_boundary_skip_w0015_v1")


def hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / "code").rglob("*.py"))}


def gate(path, source):
    record = read_json(path)
    if record.get("status") != "PASS" or record.get("source_hashes") != source:
        raise RuntimeError(f"Missing or stale verification: {path}")
    return record


def gpu_free():
    result = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    return {int(p[0]): dict(free_mib=int(p[1]), utilization=int(p[2]))
            for p in (line.split(",") for line in result.splitlines())}


def prepare_jobs(phase, run_dir, smoke):
    jobs = []
    for dataset, labels in (("LA", 8), ("Pancreas", 6)):
        name = f"{dataset}_{labels}l"
        baseline = PROJECT / "Results/seed_42" / name
        config = read_json(baseline / "config.json")
        manifest = read_json(baseline / "data_split_manifest.json")
        if not config or config["seed"] != 42 or config["max_iterations"] != 15000:
            raise RuntimeError(f"Invalid Baseline: {baseline}")
        for row in manifest["splits"].values():
            if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError("Case lists changed since Baseline")
        for variant, condition in CONDITIONS.items():
            output = ((PROJECT / "Results" / condition / "seed_42" / name) if phase == "full" else
                      run_dir / "runs" / condition / name)
            if output.exists():
                raise FileExistsError(f"Refusing duplicate/overwrite: {output}")
            command = [sys.executable, "-u", str(PROJECT / "code/train_boundary_probe.py"),
                       "--variant", variant, "--dataset", dataset, "--labelnum", str(labels),
                       "--seed", "42", "--data_root", config["data_root"], "--output_dir", str(output),
                       "--checkpoint_interval", "100", "--random_repeats", "3"]
            for key in ("batch_size", "labeled_bs", "base_lr", "ema_decay", "consistency", "num_workers", "cpu_threads"):
                command += [f"--{key}", str(config[key])]
            command += ["--patch_size", *map(str, config["patch_size"])]
            if phase == "smoke":
                settings = dict(max_iterations=8, warmup_iterations=2, val_interval=0,
                                checkpoint_interval=4, log_interval=1, gradient_interval=2)
                if variant == "d3":
                    settings.update(boundary_start=2, boundary_ramp=2)
                required = 9000
            else:
                settings = dict(max_iterations=15000, warmup_iterations=2000, val_start=800,
                                val_interval=200, log_interval=10, gradient_interval=200)
                measured = next(r for r in smoke["runs"] if r["variant"] == variant and r["dataset"] == dataset)
                required = math.ceil(measured["peak_reserved_mib"] + 1024)
            for key, value in settings.items():
                command += [f"--{key}", str(value)]
            jobs.append(dict(variant=variant, condition=condition, dataset=dataset, labelnum=labels,
                             seed=42, command=command, output_dir=str(output), baseline_dir=str(baseline),
                             launch_log=str(run_dir / f"{variant}_{name}.log"),
                             state="queued", gpu=None, pid=None, exit_code=None,
                             required_free_mib=required, initialization="random_seed_no_checkpoint"))
    return jobs


def snapshot(manifest, run_dir):
    rows = []
    for run in manifest["runs"]:
        output = Path(run["output_dir"])
        status = read_json(output / "run_status.json")
        best = read_json(output / "best_metrics.json")
        baseline = read_json(Path(run["baseline_dir"]) / "best_metrics.json")
        if run["state"] not in ("queued", "failed", "completed"):
            run["state"] = status.get("state", "initializing")
        rows.append(dict(condition=run["condition"], variant=run["variant"], dataset=run["dataset"],
                         state=run["state"], gpu=run["gpu"], pid=run["pid"],
                         iteration=status.get("iteration", 0), best_dice=best.get("dice"),
                         baseline_dice=baseline.get("dice"), output_dir=run["output_dir"],
                         required_free_mib=run["required_free_mib"], exit_code=run["exit_code"]))
    manifest["updated_at"] = stamp()
    write_json(run_dir / "launch_manifest.json", manifest)
    destination = COMPARISON if manifest["phase"] == "full" else run_dir
    write_json(destination / "full_launch_manifest.json", manifest)
    for variant in (None, *CONDITIONS):
        selected = rows if variant is None else [r for r in rows if r["variant"] == variant]
        target = destination if variant is None else ((PROJECT / "Results" / CONDITIONS[variant])
                                                       if manifest["phase"] == "full" else run_dir / CONDITIONS[variant])
        write_json(target / "summary.json", dict(updated_at=stamp(), phase=manifest["phase"],
                   state=manifest["state"], protocol="legacy_test_selection_exploratory", rows=selected))
        if variant is not None:
            write_json(target / "full_launch_manifest.json", dict(manifest, runs=[r for r in manifest["runs"] if r["variant"] == variant]))
        lines = ["# E4 / D3 boundary experiments", "", f"Updated: {stamp()}", "",
                 "| Condition | Dataset | GPU | State | Step | Best Dice (%) |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for row in selected:
            best = "pending" if row["best_dice"] is None else f"{100 * row['best_dice']:.4f}"
            lines.append(f"| {row['variant']} | {row['dataset']} 10% | {row['gpu']} | {row['state']} | {row['iteration']} | {best} |")
        (target / "summary.md").write_text("\n".join(lines) + "\n")


def check_smoke(manifest, source):
    import torch
    sys.path.insert(0, str(PROJECT / "code"))
    from networks.vnet_boundary_probe import VNetBoundaryProbe
    results = []
    for run in manifest["runs"]:
        path = Path(run["output_dir"])
        status = read_json(path / "run_status.json")
        if run["exit_code"] != 0 or status.get("state") != "completed" or status.get("iteration") != 8:
            raise RuntimeError(f"Smoke incomplete: {run['variant']} {run['dataset']}")
        with (path / "training_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 8 or not any(float(r["boundary_weight"]) > 0 for r in rows):
            raise RuntimeError("Active boundary term not exercised")
        probes = [json.loads(line) for line in (path / "channel_interventions.jsonl").read_text().splitlines()]
        if len(probes) != 5 or not all(r["score_norm"] > 0 for r in probes):
            raise RuntimeError("Missing/nonfinite online intervention diagnostics")
        if read_json(path / "data_split_manifest.json") != read_json(Path(run["baseline_dir"]) / "data_split_manifest.json"):
            raise RuntimeError("Data split changed")
        checkpoint = torch.load(path / "last_checkpoint.pth", map_location="cpu", weights_only=False)
        model = VNetBoundaryProbe(1, 2, normalization="batchnorm", has_dropout=True, variant=run["variant"])
        model.load_state_dict(checkpoint["student"])
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint["optimizer"])
        assert checkpoint["iteration"] == 8
        assert optimizer.state[model.boundary_head[0].weight]["momentum_buffer"].norm() > 0
        results.append(dict(variant=run["variant"], dataset=run["dataset"], output_dir=str(path),
                            final_loss=float(rows[-1]["loss_total"]), diagnostics=len(probes),
                            peak_memory_mib=max(float(r["peak_memory_mib"]) for r in rows),
                            peak_reserved_mib=max(float(r["peak_reserved_mib"]) for r in rows)))
        del checkpoint, optimizer, model
    write_json(VERIFY / "smoke_results.json", dict(status="PASS", checked_at=stamp(), source_hashes=source, runs=results))
    print(json.dumps(dict(status="PASS", runs=results), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "full"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument("--max_concurrent", type=int, default=4)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--run_dir", type=Path)
    args = parser.parse_args()
    source = hashes()
    gate(VERIFY / "unit_results.json", source)
    smoke = gate(VERIFY / "smoke_results.json", source) if args.phase == "full" else None
    suffix = stamp().replace(":", "").replace("+", "_")
    run_dir = args.run_dir or ((VERIFY / f"smoke_{suffix}") if args.phase == "smoke" else
                              PROJECT / "launch_logs" / f"E4_D3_{suffix}")
    jobs = prepare_jobs(args.phase, run_dir, smoke)
    if args.check_only:
        print(json.dumps(dict(jobs=jobs, gpu_status=gpu_free()), indent=2))
        return
    destination = COMPARISON if args.phase == "full" else run_dir
    if (destination / "full_launch_manifest.json").exists():
        raise FileExistsError("Already launched; inspect existing manifest")
    run_dir.mkdir(parents=True, exist_ok=args.run_dir is not None)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", args.phase,
                   "--gpus", *map(str, args.gpus), "--max_concurrent", str(args.max_concurrent),
                   "--run_dir", str(run_dir)]
        with (run_dir / "supervisor.log").open("x") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=process.pid, started_at=stamp(), log_dir=str(run_dir), command=command)
        write_json(destination / "supervisor.json", record)
        print(json.dumps(record, indent=2), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(phase=args.phase, state="running", started_at=stamp(), supervisor_pid=os.getpid(),
                        runs=jobs, source_hashes=source, allowed_gpus=args.gpus,
                        scheduler="one_own_job_per_GPU_free_memory_at_least_smoke_peak_reserved_plus1024MiB",
                        source_launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        children = {}
        environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                           PYTHONUNBUFFERED="1")
        while True:
            for i, child in list(children.items()):
                code = child.poll()
                if code is not None:
                    run = jobs[i]
                    run["exit_code"] = code
                    status = read_json(Path(run["output_dir"]) / "run_status.json")
                    run["state"] = "completed" if code == 0 and status.get("state") == "completed" else "failed"
                    del children[i]
            free = gpu_free()
            active = {jobs[i]["gpu"] for i in children}
            for i, run in enumerate(jobs):
                if run["state"] != "queued" or len(children) >= args.max_concurrent:
                    continue
                candidates = [g for g in args.gpus if g not in active and
                              free.get(g, {}).get("free_mib", 0) >= run["required_free_mib"]]
                if not candidates:
                    continue
                if hashes() != source:
                    raise RuntimeError("Training code changed after verification; refusing queued launch")
                gpu = max(candidates, key=lambda g: free[g]["free_mib"])
                command = run["command"] + ["--gpu", str(gpu)]
                with Path(run["launch_log"]).open("x") as log:
                    child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                run.update(gpu=gpu, pid=child.pid, command=command, state="initializing", started_at=stamp())
                children[i] = child
                active.add(gpu)
                print(f"Launched {run['variant']} {run['dataset']} 10%: GPU{gpu}, PID{child.pid}", flush=True)
            if not children and all(r["state"] in ("completed", "failed") for r in jobs):
                manifest["state"] = "completed" if all(r["state"] == "completed" for r in jobs) else "failed"
            snapshot(manifest, run_dir)
            if manifest["state"] != "running":
                break
            time.sleep(3 if args.phase == "smoke" else 30)
        if manifest["state"] != "completed":
            raise RuntimeError("One or more jobs failed; inspect per-run logs")
        if args.phase == "smoke":
            check_smoke(manifest, source)


if __name__ == "__main__":
    main()
