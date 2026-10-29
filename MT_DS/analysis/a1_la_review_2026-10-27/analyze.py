"""Read-only analysis of existing training records; no model loads or training."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics as stats

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
OUT = Path(__file__).resolve().parent
A1 = PROJECT / "Results/A1_boundary_supervision_v1/seed_42/LA_8l"
BASE = ROOT / "MT_DS/Results/seed_42/LA_8l"
OLD = ROOT / "MT+DuCiSC+BGCE/MT-DuCiSC/results"


def js(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def csv_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def average(rows, key):
    return stats.mean(float(row[key]) for row in rows)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    a, b = js(A1 / "best_metrics.json"), js(BASE / "best_metrics.json")
    lookup = {row["case"]: row for row in b["per_case"]}
    deltas = []
    for row in a["per_case"]:
        record = dict(case=row["case"])
        for metric in ("dice", "jc", "hd95", "asd"):
            record[f"baseline_{metric}"] = lookup[row["case"]][metric]
            record[f"a1_{metric}"] = row[metric]
            record[f"delta_{metric}"] = row[metric] - lookup[row["case"]][metric]
        deltas.append(record)
    write_csv(OUT / "case_deltas.csv", deltas)
    metrics = {}
    for metric in ("dice", "jc", "hd95", "asd"):
        direction = 1 if metric in ("dice", "jc") else -1
        metrics[metric] = dict(baseline=b[metric], a1=a[metric], delta=a[metric] - b[metric],
                               improved_cases=sum(direction * row[f"delta_{metric}"] > 0 for row in deltas),
                               worsened_cases=sum(direction * row[f"delta_{metric}"] < 0 for row in deltas),
                               baseline_median=stats.median(row[metric] for row in b["per_case"]),
                               a1_median=stats.median(row[metric] for row in a["per_case"]))
    va = {row["iteration"]: row for row in jsonl(A1 / "validation_metrics.jsonl")}
    vb = {row["iteration"]: row for row in jsonl(BASE / "validation_metrics.jsonl")}
    curves = []
    for step in sorted(va.keys() & vb.keys()):
        row = dict(iteration=step)
        for metric in ("dice", "hd95", "asd"):
            row[f"baseline_{metric}"] = vb[step][metric]
            row[f"a1_{metric}"] = va[step][metric]
            row[f"delta_{metric}"] = va[step][metric] - vb[step][metric]
        curves.append(row)
    write_csv(OUT / "matched_validation_curves.csv", curves)
    ta = {int(row["iteration"]): row for row in csv_rows(A1 / "training_metrics.csv")}
    tb = {int(row["iteration"]): row for row in csv_rows(BASE / "training_metrics.csv")}
    probes = jsonl(A1 / "boundary_gradient_metrics.jsonl")
    phases = []
    for lo, hi in ((1, 2000), (2001, 4000), (4001, 8000), (8001, 12000), (12001, 15000)):
        rows = [row for step, row in ta.items() if lo <= step <= hi]
        phase = dict(first_step=lo, last_step=hi,
                     training={key: average(rows, key) for key in (
                         "loss_supervised", "loss_consistency", "loss_boundary", "boundary_half",
                         "boundary_full", "boundary_band_fraction")},
                     baseline_supervised=average([row for step, row in tb.items() if lo <= step <= hi],
                                                 "loss_supervised"), gradients={})
        records = [row for row in probes if lo <= row["iteration"] <= hi]
        for scope in ("combined_observed_parameters", "half_upsampling", "final_conv"):
            selected = records if scope == "combined_observed_parameters" else [row["parameters"][scope] for row in records]
            phase["gradients"][scope] = dict(
                count=len(selected), median_cosine=stats.median(row["cosine"] for row in selected),
                negative_cosines=sum(row["cosine"] < 0 for row in selected),
                median_weighted_ratio=stats.median(row["weighted_ratio"] for row in selected),
                max_weighted_ratio=max(row["weighted_ratio"] for row in selected),
                regional_gradient_projection_reversals=sum(
                    1 + row["cosine"] * row["weighted_ratio"] < 0 for row in selected))
        phases.append(phase)
    old_runs = []
    for log in sorted(OLD.rglob("log.txt")):
        if "/code/" in str(log):
            continue
        directory = log.parent
        lines = log.read_text(errors="replace").splitlines()
        if not lines or "Namespace(" not in lines[0]:
            continue
        header = lines[0][lines[0].index("Namespace("):]
        try:
            call = ast.parse(header, mode="eval").body
            config = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
        except (ValueError, SyntaxError):
            continue
        if config.get("dataset_name") not in ("LA", "Pancreas"):
            continue
        validations = []
        for line in lines:
            match = re.search(r"iter (\d+), dice_sample: \[([\d.]+)/([\d.]+)\]", line)
            if match:
                validations.append(dict(iteration=int(match[1]), dice_percent=float(match[2]),
                                        running_best_percent=float(match[3])))
        performance = {}
        for path in sorted(directory.glob("*performance_0.txt")):
            if "chunk" in path.name:
                continue
            last = path.read_text(errors="replace").strip().splitlines()[-1]
            values = re.search(r"dice: ([\d.]+).*jc: ([\d.]+).*hd: ([\d.]+).*asd: ([\d.]+)", last)
            if values:
                performance[path.name] = dict(zip(("dice_percent", "jc_percent", "hd95", "asd"),
                                                  map(float, values.groups())))
        sources = {}
        for rel in ("train_3D.py", "networks/VNet.py", "networks/boundary_probe.py", "utils/losses.py", "utils/test_patch.py"):
            path = directory / "code" / rel
            if path.exists():
                sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        old_runs.append(dict(directory=str(directory), config=config, performance=performance,
                             validation_count=len(validations),
                             best_logged_dice_percent=max((v["dice_percent"] for v in validations), default=None),
                             final_logged=validations[-1] if validations else None, source_hashes=sources))

    # Label-target diagnostic: two targets share a coarse cell at the half head.
    target_stats = []
    cache = Path(js(A1 / "config.json")["boundary_cache_manifest"]).parent
    for path in sorted(cache.glob("*.npy")):
        band = np.load(path, mmap_mode="r")
        even = tuple(n - n % 2 for n in band.shape)
        values = np.asarray(band[tuple(slice(0, n) for n in even)])
        block_shape = tuple(v for n in even for v in (n // 2, 2))
        inside = (values == 1).reshape(block_shape).sum(axis=(1, 3, 5)).astype(np.float64)
        outside = (values == 2).reshape(block_shape).sum(axis=(1, 3, 5)).astype(np.float64)
        active = inside + outside > 0
        mixed = (inside > 0) & (outside > 0)
        ia = inside / max(inside.sum(), 1) / 2
        oa = outside / max(outside.sum(), 1) / 2
        probability = ia / np.maximum(ia + oa, 1e-300)
        minimum = -(ia * np.log(np.clip(probability, 1e-300, 1))
                    + oa * np.log(np.clip(1 - probability, 1e-300, 1))).sum()
        target_stats.append(dict(cache_file=path.name, analyzed_shape=even,
                                 mixed_fraction_of_boundary_cells=float(mixed.sum() / active.sum()),
                                 band_mass_fraction_in_mixed_cells=float((inside[mixed].sum() + outside[mixed].sum()) /
                                                                        (inside.sum() + outside.sum())),
                                 unconstrained_half_head_minimum_CE_full_volume=float(minimum)))
    warmup = {key: max(abs(float(ta[i][key]) - float(tb[i][key])) for i in ta.keys() & tb.keys() if i <= 2000)
              for key in ("loss_supervised", "ce", "dice", "loss_consistency", "gradient_norm")}
    plateau_probes = [row for row in probes if row["iteration"] >= 4000]
    plateau = dict(first_step=4000, last_step=15000, count=len(plateau_probes),
                   scope="Concatenation of the two observed parameter tensors; not the whole model",
                   median_weighted_ratio=stats.median(row["weighted_ratio"] for row in plateau_probes),
                   ratios_above_one=sum(row["weighted_ratio"] > 1 for row in plateau_probes),
                   median_cosine=stats.median(row["cosine"] for row in plateau_probes),
                   negative_cosines=sum(row["cosine"] < 0 for row in plateau_probes))
    report = dict(scope="Existing records and labels only; no training, model load, inference, or result overwrite",
                  available_result="A1 best at 14800; final 15000 validation/test missing",
                  baseline_best_iteration=b["iteration"], a1_best_iteration=a["iteration"],
                  metrics=metrics, case_deltas=deltas, matched_validation_count=len(curves),
                  matched_improvements={metric: sum((row[f"delta_{metric}"] > 0 if metric == "dice"
                                                    else row[f"delta_{metric}"] < 0) for row in curves)
                                        for metric in ("dice", "hd95", "asd")},
                  warmup_max_abs_differences=warmup, phases=phases, plateau_observed_gradients=plateau,
                  old_runs=old_runs,
                  coarse_target_scope="Full-volume labels trimmed to even shapes and aligned to their origin. "
                  "This illustrates target mixing; it is not a measurement on actual augmented training crops "
                  "and does not establish a cause of held-out performance changes.",
                  coarse_target_diagnostics=target_stats)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    steps = [row["iteration"] for row in curves]
    for axis, metric, factor, title in ((axes[0, 0], "dice", 100, "LA Dice (%)"),
                                        (axes[0, 1], "hd95", 1, "LA HD95 (voxel)")):
        axis.plot(steps, [factor * row[f"baseline_{metric}"] for row in curves], label="MT_DS")
        axis.plot(steps, [factor * row[f"a1_{metric}"] for row in curves], label="A1")
        axis.set_title(title)
        axis.legend()
    for scope, label in (("half_upsampling", "Half upsampling"), ("final_conv", "Final conv")):
        axes[1, 0].plot([row["iteration"] for row in probes],
                        [row["parameters"][scope]["weighted_ratio"] for row in probes], label=label)
        axes[1, 1].plot([row["iteration"] for row in probes],
                        [row["parameters"][scope]["cosine"] for row in probes], label=label)
    axes[1, 0].axhline(1, color="black", linestyle=":", linewidth=1)
    axes[1, 0].set_title("Weighted boundary / region gradient norm (observed blocks)")
    axes[1, 1].axhline(0, color="black", linestyle=":", linewidth=1)
    axes[1, 1].set_title("Boundary-region gradient cosine (observed blocks)")
    for axis in axes.flat:
        axis.axvline(2000, color="gray", linestyle="--", linewidth=0.8)
        axis.axvline(4000, color="gray", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.2)
    axes[1, 0].legend()
    axes[1, 1].legend()
    fig.savefig(OUT / "training_evidence.png", dpi=180)
    fig.savefig(OUT / "training_evidence.pdf")
    print(json.dumps(dict(output_dir=str(OUT), metrics=metrics, warmup=warmup,
                          coarse_target_diagnostics=target_stats), indent=2))


if __name__ == "__main__":
    main()
