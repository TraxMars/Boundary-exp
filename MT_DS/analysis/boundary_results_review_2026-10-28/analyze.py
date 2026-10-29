"""Review existing training records only; no model loads, inference, or training."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONDITIONS = {
    "baseline": PROJECT / "Results",
    "band_0.1": PROJECT / "Results/A1_boundary_supervision_v1",
    "band_0.03": PROJECT / "Results/A1_boundary_supervision_w003_v1",
    "head_0.03": PROJECT / "Results/A1_independent_boundary_head_w003_v1",
}
DATASETS = ("LA_8l", "Pancreas_6l")
METRICS = ("dice", "jc", "hd95", "asd")
PHASES = ((1, 1300), (1301, 2000), (2001, 4000), (4001, 8000), (8001, 12000), (12001, 15000))


def js(path):
    return json.loads(path.read_text()) if path.exists() else {}


def jl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save_csv(name, records):
    keys = list(dict.fromkeys(k for record in records for k in record))
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def avg(values):
    values = [v for v in values if v is not None]
    return st.mean(values) if values else None


def compare_cases(before, after):
    lookup = {r["case"]: r for r in before["per_case"]}
    assert set(lookup) == {r["case"] for r in after["per_case"]}
    records = []
    for row in after["per_case"]:
        record = dict(case=row["case"])
        for metric in METRICS:
            a, b = lookup[row["case"]][metric], row[metric]
            record.update({f"before_{metric}": a, f"after_{metric}": b,
                           f"delta_{metric}": b - a if a is not None and b is not None else None})
        records.append(record)
    return records


def case_stats(records):
    result = {}
    for metric in METRICS:
        delta = [r[f"delta_{metric}"] for r in records if r[f"delta_{metric}"] is not None]
        direction = 1 if metric in ("dice", "jc") else -1
        result[metric] = dict(valid_pairs=len(delta), mean_delta=avg(delta), median_delta=st.median(delta),
                              improved=sum(direction * x > 0 for x in delta),
                              worsened=sum(direction * x < 0 for x in delta),
                              median_before=st.median(r[f"before_{metric}"] for r in records),
                              median_after=st.median(r[f"after_{metric}"] for r in records))
    return result


def gradient_stats(records):
    result = {}
    for scope in ("combined_observed_parameters", "half_upsampling", "final_conv"):
        selected = records if scope == "combined_observed_parameters" else [r["parameters"][scope] for r in records]
        selected = [r for r in selected if r["cosine"] is not None and r["weighted_ratio"] is not None]
        if not selected:
            continue
        ratio = [r["weighted_ratio"] for r in selected]
        cosine = [r["cosine"] for r in selected]
        result[scope] = dict(count=len(selected), median_weighted_ratio=st.median(ratio),
                             ratio_q25=float(np.quantile(ratio, .25)), ratio_q75=float(np.quantile(ratio, .75)),
                             max_ratio=max(ratio), ratios_above_one=sum(x > 1 for x in ratio),
                             median_cosine=st.median(cosine), negative_cosines=sum(x < 0 for x in cosine),
                             negative_projection_with_region=sum(1 + r * c < 0 for r, c in zip(ratio, cosine)))
    return result


def main():
    loaded, metrics_table, cases_table, curve_table, phase_table, gradient_table = {}, [], [], [], [], []
    report = dict(scope="Existing saved metrics, training logs and source/configuration manifests only",
                  limitations=["Single seed 42; test split reused for selection",
                               "Distance units: voxels; ASD is directed, not ASSD",
                               "Samplers not explicitly serialized on resume; epoch-boundary/trace checks recorded below",
                               "Band 0.1 LA missing final validation/test; best through 14800 only",
                               "Gradient observations cover two parameter tensors; exclude MT gradient and momentum"],
                  datasets={})
    for dataset in DATASETS:
        runs = {}
        for condition, directory in CONDITIONS.items():
            path = directory / "seed_42" / dataset
            validation = jl(path / "validation_metrics.jsonl")
            training = rows(path / "training_metrics.csv")
            assert len({r["iteration"] for r in validation}) == len(validation)
            assert len({r["iteration"] for r in training}) == len(training)
            runs[condition] = dict(path=path, config=js(path / "config.json"), status=js(path / "run_status.json"),
                                   best=js(path / "best_metrics.json"), test=js(path / "test_metrics.json"),
                                   val={r["iteration"]: r for r in validation},
                                   train={int(r["iteration"]): r for r in training},
                                   gradients=jl(path / "boundary_gradient_metrics.jsonl"))
        loaded[dataset] = runs
        base = runs["baseline"]
        dataset_report = {}
        all_steps = sorted(set().union(*(set(run["val"]) for run in runs.values())))
        for step in all_steps:
            record = dict(dataset=dataset, iteration=step)
            for condition, run in runs.items():
                for metric in METRICS:
                    record[f"{condition}_{metric}"] = run["val"].get(step, {}).get(metric)
            curve_table.append(record)
        for condition, run in runs.items():
            path, best = run["path"], run["best"]
            assert best["cases"] == 20
            for metric in METRICS:
                observed = [r[metric] for r in best["per_case"] if r[metric] is not None]
                assert abs(st.mean(observed) - best[metric]) < 1e-12
            if run["test"]:
                for metric in METRICS:
                    assert run["test"][metric] == best[metric]
            for filename in ("data_split_manifest.json",):
                assert js(path / filename) == js(base["path"] / filename)
            frozen = ("code/evaluate.py", "code/objectives.py", "code/networks/vnet_ds.py")
            source_checks = {rel: hashlib.sha256((path / rel).read_bytes()).hexdigest() ==
                            hashlib.sha256((base["path"] / rel).read_bytes()).hexdigest() for rel in frozen}
            assert all(source_checks.values())
            if condition != "baseline":
                assert js(path / "boundary_target_manifest.json") == js(runs["band_0.1"]["path"] / "boundary_target_manifest.json")
            case_records = compare_cases(base["best"], best)
            for record in case_records:
                cases_table.append(dict(dataset=dataset, condition=condition, **record))
            last = run["val"][max(run["val"])]
            summary = dict(dataset=dataset, condition=condition, state=run["status"]["state"],
                           updates=run["status"]["iteration"], best_iteration=best["iteration"],
                           best_retest_exists=bool(run["test"]), validation_count=len(run["val"]),
                           surface_metric_cases=best["surface_metric_cases"])
            for metric in METRICS:
                summary[f"best_{metric}"] = best[metric]
                summary[f"delta_best_{metric}"] = best[metric] - base["best"][metric]
                summary[f"last_{metric}"] = last[metric]
            metrics_table.append(summary)
            audit = dict(directory=str(path), summary=summary, last_validation_iteration=last["iteration"],
                         source_checks=source_checks, split_identical_to_baseline=True,
                         resumed=sorted(str(p) for p in path.glob("resume_history/*/*record.json")),
                         case_comparison=case_stats(case_records),
                         worst_dice_cases=sorted(case_records, key=lambda r:r["delta_dice"])[:3],
                         worst_hd95_cases=sorted(case_records, key=lambda r:r["delta_hd95"], reverse=True)[:3])
            audit["resume_events"] = []
            for record_path in path.glob("resume_history/*/*record.json"):
                event = js(record_path)
                epoch_steps = run["config"]["labelnum"] // run["config"]["labeled_bs"]
                audit["resume_events"].append(dict(checkpoint_step=event["checkpoint_iteration"],
                    epoch_steps=epoch_steps, at_epoch_boundary=event["checkpoint_iteration"] % epoch_steps == 0,
                    note=event.get("note")))
            if condition != "baseline":
                ref = runs["band_0.1"]["train"]
                common = sorted(run["train"].keys() & ref.keys())
                audit["logged_sampling_trace_vs_band_01"] = dict(count=len(common),
                    band_fraction_max_abs_delta=max(abs(float(run["train"][i]["boundary_band_fraction"])-float(ref[i]["boundary_band_fraction"])) for i in common),
                    empty_fraction_max_abs_delta=max(abs(float(run["train"][i]["empty_foreground_fraction"])-float(ref[i]["empty_foreground_fraction"])) for i in common))
                for scale in ("half", "full"):
                    delta = [abs(float(run["train"][i][f"boundary_{scale}_positive_mass"])-
                                 float(ref[i][f"boundary_{scale}_inner_mass"])-float(ref[i][f"boundary_{scale}_outer_mass"]))
                             if condition == "head_0.03" else max(abs(float(run["train"][i][f"boundary_{scale}_{side}_mass"])-float(ref[i][f"boundary_{scale}_{side}_mass"])) for side in ("inner", "outer"))
                             for i in common]
                    audit["logged_sampling_trace_vs_band_01"][f"{scale}_target_mass_max_abs_delta"] = max(delta)
            protocol_keys = ("seed", "batch_size", "labeled_bs", "max_iterations", "warmup_iterations",
                             "base_lr", "ema_decay", "consistency", "num_workers", "cpu_threads", "patch_size",
                             "val_interval", "val_start", "normalization", "teacher_bn", "consistency_loss")
            audit["protocol_differences_vs_baseline"] = {key: [base["config"].get(key), run["config"].get(key)]
                                                         for key in protocol_keys if base["config"].get(key) != run["config"].get(key)}
            assert not audit["protocol_differences_vs_baseline"]
            audit["warmup"] = {}
            for lo, hi in ((1, 1300), (1301, 2000)):
                common = [i for i in run["train"].keys() & base["train"].keys() if lo <= i <= hi]
                audit["warmup"][f"{lo}_{hi}"] = {key: max(abs(float(run["train"][i][key]) - float(base["train"][i][key])) for i in common)
                                                     for key in ("loss_supervised", "loss_consistency", "gradient_norm")}
            audit["training_phases"], audit["validation_phases"] = [], []
            for lo, hi in PHASES:
                selected = [r for i, r in run["train"].items() if lo <= i <= hi]
                keys = ("loss_supervised", "ce", "dice", "loss_consistency", "loss_total", "gradient_norm", "loss_boundary",
                        "boundary_half", "boundary_full", "boundary_half_bce", "boundary_half_dice",
                        "boundary_full_bce", "boundary_full_dice", "boundary_band_fraction")
                phase = dict(first_step=lo, last_step=hi, count=len(selected))
                for key in keys:
                    if key in selected[0]:
                        phase[key] = st.mean(float(r[key]) for r in selected)
                if "loss_boundary" in selected[0]:
                    phase["weighted_boundary"] = st.mean(float(r["loss_boundary"]) * float(r["boundary_weight"]) for r in selected)
                audit["training_phases"].append(phase)
                phase_table.append(dict(dataset=dataset, condition=condition, **phase))
                selected_steps = [s for s in run["val"].keys() & base["val"].keys() if lo <= s <= hi]
                if selected_steps:
                    entry = dict(first_step=lo, last_step=hi, count=len(selected_steps))
                    for metric in METRICS:
                        delta = [run["val"][s][metric] - base["val"][s][metric] for s in selected_steps]
                        entry[metric] = dict(mean=avg([run["val"][s][metric] for s in selected_steps]),
                                              mean_delta=avg(delta),
                                              median=st.median(run["val"][s][metric] for s in selected_steps),
                                              median_delta=st.median(delta),
                                              better_count=sum((d > 0 if metric in ("dice", "jc") else d < 0) for d in delta))
                    audit["validation_phases"].append(entry)
            common = sorted(s for s in run["val"].keys() & base["val"].keys() if s >= 4000)
            audit["matched_after_4000"] = {metric: dict(count=len(common),
                mean_delta=st.mean(run["val"][s][metric] - base["val"][s][metric] for s in common),
                improved=sum((run["val"][s][metric] > base["val"][s][metric] if metric in ("dice", "jc") else
                              run["val"][s][metric] < base["val"][s][metric]) for s in common)) for metric in METRICS}
            audit["post_4000_dice_median"] = st.median(run["val"][s]["dice"] for s in common)
            worst = sorted((run["val"][s] for s in common), key=lambda r:r["dice"])[:3]
            audit["worst_evaluations_after_4000"] = []
            for record in worst:
                step = record["iteration"]
                entry = {key:record[key] for key in ("iteration", "dice", "hd95", "asd", "surface_metric_cases")}
                entry["training_at_step"] = {key:float(run["train"][step][key]) for key in ("loss_supervised", "loss_consistency", "gradient_norm")}
                entry["adjacent_evaluations"] = [{key:run["val"][s][key] for key in ("iteration", "dice", "hd95", "asd")} for s in (step-200, step+200) if s in run["val"]]
                audit["worst_evaluations_after_4000"].append(entry)
            late_steps = [s for s in run["val"] if s >= 12000]
            audit["late_validation"] = {metric: dict(count=len(late_steps), mean=avg([run["val"][s][metric] for s in late_steps]),
                                                     sd=st.stdev(run["val"][s][metric] for s in late_steps)) for metric in METRICS}
            audit["minimum_surface_metric_cases_in_curve"] = min(r["surface_metric_cases"] for r in run["val"].values())
            if run["gradients"]:
                for lo, hi in ((2001, 4000), (4001, 8000), (8001, 12000), (12001, 15000), (4000, 15000)):
                    stats = gradient_stats([r for r in run["gradients"] if lo <= r["iteration"] <= hi])
                    audit.setdefault("gradient_phases", {})[f"{lo}_{hi}"] = stats
                    for scope, values in stats.items():
                        gradient_table.append(dict(dataset=dataset, condition=condition, first_step=lo,
                                                   last_step=hi, scope=scope, **values))
            dataset_report[condition] = audit
        report["datasets"][dataset] = dataset_report
    pairs = []
    for dataset, runs in loaded.items():
        for before, after in (("band_0.1", "band_0.03"), ("band_0.03", "head_0.03")):
            records = compare_cases(runs[before]["best"], runs[after]["best"])
            report["datasets"][dataset].setdefault("pairwise", {})[f"{before}_to_{after}"] = case_stats(records)
            pairs.extend(dict(dataset=dataset, before=before, after=after, **r) for r in records)
    save_csv("best_and_last_metrics.csv", metrics_table)
    save_csv("case_deltas.csv", cases_table)
    save_csv("pairwise_case_deltas.csv", pairs)
    save_csv("validation_curves.csv", curve_table)
    save_csv("training_phases.csv", phase_table)
    save_csv("gradient_statistics.csv", gradient_table)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(baseline="#252525", **{"band_0.1":"#d95f02", "band_0.03":"#7570b3", "head_0.03":"#1b9e77"})
    labels = dict(baseline="MT_DS", **{"band_0.1":"Band CE 0.1", "band_0.03":"Band CE 0.03", "head_0.03":"Independent heads 0.03"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(("dice", "hd95", "asd")):
            ax = axes[i,j]
            for condition, run in loaded[dataset].items():
                steps = sorted(run["val"])
                ax.plot(steps, [run["val"][s][metric]*(100 if metric=="dice" else 1) for s in steps],
                        color=colors[condition], label=labels[condition], linewidth=1.3)
            ax.set_title(f"{dataset.split('_')[0]}: {metric.upper()} ({'%' if metric=='dice' else 'voxel'})")
            ax.set_xlabel("Optimizer updates")
            ax.axvline(2000,color="gray",linestyle="--",linewidth=.8)
            ax.axvline(4000,color="gray",linestyle="--",linewidth=.8)
            ax.grid(alpha=.2)
            if j==0: ax.legend(fontsize=8)
    fig.savefig(OUT / "validation_curves.png", dpi=180)
    fig.savefig(OUT / "validation_curves.pdf")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(("dice", "hd95", "asd")):
            ax = axes[i,j]
            for condition, run in loaded[dataset].items():
                steps = sorted(s for s in run["val"] if s >= 12000)
                ax.plot(steps, [run["val"][s][metric]*(100 if metric=="dice" else 1) for s in steps],
                        color=colors[condition], label=labels[condition], linewidth=1.3, marker=".")
            ax.set_title(f"{dataset.split('_')[0]}: late {metric.upper()} ({'%' if metric=='dice' else 'voxel'})")
            ax.set_xlabel("Optimizer updates")
            ax.grid(alpha=.2)
            if j==0: ax.legend(fontsize=8)
    fig.savefig(OUT / "late_validation_curves.png", dpi=180)
    fig.savefig(OUT / "late_validation_curves.pdf")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.3), constrained_layout=True)
    for i, dataset in enumerate(DATASETS):
        for condition, run in loaded[dataset].items():
            train = sorted(run["train"].items())
            steps = np.array([s for s,r in train])
            values = np.array([float(r["loss_supervised"]) for s,r in train])
            width=50
            axes[i,0].plot(steps[width-1:], np.convolve(values,np.ones(width)/width,mode="valid"),
                           label=labels[condition],color=colors[condition])
            if run["gradients"]:
                records = [r for r in run["gradients"] if r["iteration"]>=2000]
                axes[i,1].plot([r["iteration"] for r in records],[r["weighted_ratio"] for r in records],
                               label=labels[condition],color=colors[condition],alpha=.85,linewidth=1)
                axes[i,2].plot([r["iteration"] for r in records],[r["cosine"] for r in records],
                               label=labels[condition],color=colors[condition],alpha=.85,linewidth=1)
        axes[i,0].set_title(f"{dataset.split('_')[0]}: supervised loss, 500-update average")
        axes[i,1].set_title("Weighted boundary / region gradient norm (two tensors)")
        axes[i,1].axhline(1,color="black",linestyle=":",linewidth=1)
        axes[i,2].set_title("Boundary-region gradient cosine (two tensors)")
        axes[i,2].axhline(0,color="black",linestyle=":",linewidth=1)
        for ax in axes[i]:
            ax.axvline(2000,color="gray",linestyle="--",linewidth=.8)
            ax.axvline(4000,color="gray",linestyle="--",linewidth=.8)
            ax.set_xlabel("Optimizer updates")
            ax.grid(alpha=.2)
        axes[i,0].legend(fontsize=8)
    fig.savefig(OUT / "optimization_diagnostics.png", dpi=180)
    fig.savefig(OUT / "optimization_diagnostics.pdf")
    print(json.dumps(dict(output_dir=str(OUT),metrics=metrics_table),indent=2))


if __name__ == "__main__":
    main()
