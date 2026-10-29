"""Analyze saved training records only; no checkpoints, inference, or training."""
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

import numpy as np
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONDITIONS = dict(baseline="", band03="A1_boundary_supervision_w003_v1",
                  dual03="A1_independent_boundary_head_w003_v1",
                  e4="E4_legacy_boundary_w003_v1", d3="D3_boundary_skip_w0015_v1")
DATASETS = ("LA_8l", "Pancreas_6l")
METRICS = ("dice", "jc", "hd95", "asd")
PHASES = ((1, 2000), (2001, 4000), (4001, 8000), (8001, 12000), (12001, 15000))


def js(path):
    return json.loads(path.read_text())


def jl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(n=0)
    assert np.isfinite(values).all()
    return dict(n=len(values), mean=st.mean(values), median=st.median(values),
                min=min(values), max=max(values), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)))


def correlation(a, b):
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return None
    return float(spearmanr(a, b).statistic)


def save_csv(name, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def summarize_probes(probes, lo, hi):
    chosen = [p for p in probes if lo <= p["iteration"] <= hi and p["informative_boundary_samples"] > 0]
    result = dict(count=len(chosen), groups={})
    result["local_group_pearson"] = statistics(p["small_step_group_pearson"] for p in chosen)
    result["local_group_sign_agreement"] = statistics(p["small_step_group_sign_agreement"] for p in chosen)
    for epsilon in (.1, 1.):
        entries, predictions, actual, main_correlations, region_correlations = [], [], [], [], []
        for probe in chosen:
            rows = [r for r in probe["interventions"] if r["epsilon"] == epsilon]
            groups = {r["group"]: r for r in rows}
            random = dict(
                actual_delta={key: st.mean(groups[f"random_{j}"]["actual_delta"][key] for j in range(3))
                              for key in probe["baseline"]},
                predicted_boundary_delta=st.mean(groups[f"random_{j}"]["predicted_boundary_delta"] for j in range(3)),
                removed_feature_norm=st.mean(groups[f"random_{j}"]["removed_feature_norm"] for j in range(3)))
            groups["random_mean"] = random
            entries.append(groups)
            predictions += [r["predicted_boundary_delta"] for r in rows]
            actual += [r["actual_delta"]["boundary_loss"] for r in rows]
            main_correlations.append(correlation([r["actual_delta"]["boundary_loss"] for r in rows],
                                                 [r["actual_delta"]["final_boundary_ce"] for r in rows]))
            region_correlations.append(correlation([r["actual_delta"]["boundary_loss"] for r in rows],
                                                   [r["actual_delta"]["region_loss"] for r in rows]))
        if not entries:
            continue
        summary = dict(
            count=len(entries),
            auxiliary_vs_final_boundary_within_step_spearman=statistics(main_correlations),
            auxiliary_vs_region_within_step_spearman=statistics(region_correlations),
            top_gt_random={}, bottom_lt_random={}, groups={},
            top_random_perturbation_norm_ratio=statistics(
                r["top"]["removed_feature_norm"] / max(r["random_mean"]["removed_feature_norm"], 1e-20)
                for r in entries),
            aggregate_first_order_sign_matches=sum(p * a > 0 for p, a in zip(predictions, actual)),
            aggregate_first_order_sign_pairs=len(actual),
            top_predicted_positive=sum(r["top"]["predicted_boundary_delta"] > 0 for r in entries),
            bottom_predicted_negative=sum(r["bottom"]["predicted_boundary_delta"] < 0 for r in entries))
        valid_sign_pairs = [(p,a) for p,a in zip(predictions,actual) if abs(p)>1e-8 and abs(a)>1e-8]
        summary["valid_first_order_sign_pairs"] = len(valid_sign_pairs)
        summary["valid_first_order_sign_matches"] = sum(p*a>0 for p,a in valid_sign_pairs)
        for group in ("top", "bottom", "random_mean"):
            summary["groups"][group] = {
                key: dict(statistics(r[group]["actual_delta"][key] for r in entries),
                          positive=sum(r[group]["actual_delta"][key] > 0 for r in entries),
                          negative=sum(r[group]["actual_delta"][key] < 0 for r in entries))
                for key in chosen[0]["baseline"]}
        for key in chosen[0]["baseline"]:
            summary["top_gt_random"][key] = sum(r["top"]["actual_delta"][key] > r["random_mean"]["actual_delta"][key] for r in entries)
            summary["bottom_lt_random"][key] = sum(r["bottom"]["actual_delta"][key] < r["random_mean"]["actual_delta"][key] for r in entries)
        result["groups"][str(epsilon)] = summary
    same_batch, adjacent = [], []
    previous = None
    for p in chosen:
        top = next(r["channels"] for r in p["interventions"] if r["group"] == "top")
        same_batch.append(overlap(top[0], top[1]))
        current = set(top[0]) | set(top[1])
        if previous is not None:
            adjacent.append(overlap(current, previous))
        previous = current
    result["selection_overlap"] = dict(same_batch_two_samples=statistics(same_batch),
                                         adjacent_batch_union=statistics(adjacent),
                                         limitation="Batch cases change; no case IDs logged. Not pure temporal instability.")
    return result


def main():
    report = dict(scope="existing records only", datasets={}, limitations=[
        "One seed42 and test reused for online selection; not independent confirmation.",
        "Distances in voxel units; directed ASD, not ASSD.",
        "E4 and D3 differ in target/head/weight/schedule as well as location.",
        "Interventions freeze cached encoder features and use eval decoder before SGD.",
        "No cross-case injection or channel interaction in these conditions.",
        "Readout F1 is fixed-center-patch boundary-band F1 with different E4/D3 targets."])
    loaded, metric_rows, case_rows, phase_rows, probe_rows, readout_rows = {}, [], [], [], [], []
    curve_rows, training_rows = [], []
    for dataset in DATASETS:
        runs = {}
        for condition, relative in CONDITIONS.items():
            path = PROJECT / "Results" / relative / "seed_42" / dataset
            with (path / "training_metrics.csv").open() as stream:
                train = list(csv.DictReader(stream))
            run = dict(path=path, config=js(path / "config.json"), best=js(path / "best_metrics.json"),
                       test=js(path / "test_metrics.json"), state=js(path / "run_status.json"),
                       val=jl(path / "validation_metrics.jsonl"), train=train)
            if condition in ("e4", "d3"):
                run.update(probes=jl(path / "channel_interventions.jsonl"),
                           readout=jl(path / "boundary_validation_metrics.jsonl"))
            assert len(run["val"]) == 72 and len({r["iteration"] for r in run["val"]}) == 72
            assert run["state"]["state"] == "completed" and run["state"]["iteration"] == 15000
            assert len(train) == 1501 and len({r["iteration"] for r in train}) == 1501
            runs[condition] = run
        loaded[dataset] = runs
        base = runs["baseline"]
        base_val = {r["iteration"]: r for r in base["val"]}
        base_cases = {r["case"]: r for r in base["best"]["per_case"]}
        dataset_report = {}
        for condition, run in runs.items():
            path, best, last = run["path"], run["best"], run["val"][-1]
            assert best["dice"] == max(r["dice"] for r in run["val"])
            assert best["iteration"] == next(r["iteration"] for r in run["val"] if r["dice"] == best["dice"])
            assert js(path / "data_split_manifest.json") == js(base["path"] / "data_split_manifest.json")
            checks = {rel: digest(path / rel) == digest(base["path"] / rel)
                      for rel in ("code/evaluate.py", "code/objectives.py", "code/networks/vnet_ds.py")}
            assert all(checks.values())
            keys = ("seed", "labelnum", "batch_size", "labeled_bs", "max_iterations", "warmup_iterations",
                    "base_lr", "ema_decay", "consistency", "num_workers", "cpu_threads", "patch_size",
                    "val_interval", "val_start", "normalization", "teacher_bn", "consistency_loss")
            differences = {k: [base["config"].get(k), run["config"].get(k)]
                           for k in keys if base["config"].get(k) != run["config"].get(k)}
            assert not differences
            for metric in METRICS:
                assert best[metric] == run["test"][metric]
                assert abs(st.mean(r[metric] for r in best["per_case"]) - best[metric]) < 1e-12
            metrics = dict(dataset=dataset, condition=condition, best_step=best["iteration"],
                           final_step=last["iteration"])
            for key in METRICS:
                metrics.update({f"best_{key}": best[key], f"delta_best_{key}": best[key] - base["best"][key],
                                f"last_{key}": last[key], f"delta_last_{key}": last[key] - base["val"][-1][key]})
            metric_rows.append(metrics)
            cases = []
            for row in best["per_case"]:
                before = base_cases[row["case"]]
                r = dict(dataset=dataset, condition=condition, case=row["case"])
                for key in METRICS:
                    r.update({f"before_{key}": before[key], f"after_{key}": row[key], f"delta_{key}": row[key] - before[key]})
                cases.append(r)
                case_rows.append(r)
            case_stats = {}
            for key in METRICS:
                direction = 1 if key in ("dice", "jc") else -1
                case_stats[key] = dict(statistics(r[f"delta_{key}"] for r in cases),
                                      improved=sum(direction * r[f"delta_{key}"] > 0 for r in cases),
                                      worsened=sum(direction * r[f"delta_{key}"] < 0 for r in cases))
            audit = dict(metrics=metrics, case_stats=case_stats, source_checks=checks,
                         unchanged_main_protocol=True, unchanged_split=True,
                         best_retest_identical=True, completed_at=run["state"].get("updated_at"),
                         worst_dice_cases=sorted(cases, key=lambda r: r["delta_dice"])[:4],
                         best_dice_cases=sorted(cases, key=lambda r: r["delta_dice"], reverse=True)[:4],
                         worst_hd95_cases=sorted(cases, key=lambda r: r["delta_hd95"], reverse=True)[:3],
                         phases={})
            audit["case_contributions"] = dict(
                worst_two_dice_mean_pp=100*sum(r["delta_dice"] for r in audit["worst_dice_cases"][:2])/20,
                best_two_dice_mean_pp=100*sum(r["delta_dice"] for r in audit["best_dice_cases"][:2])/20)
            audit["incomplete_surface_evaluations"] = [
                dict(iteration=r["iteration"], surface_metric_cases=r["surface_metric_cases"],
                     cases=[c["case"] for c in r["per_case"] if c["hd95"] is None])
                for r in run["val"] if r["surface_metric_cases"] != 20]
            for r in run["val"]:
                assert r["cases"] == 20
                curve_rows.append(dict(dataset=dataset, condition=condition,
                                      surface_metric_cases=r["surface_metric_cases"],
                                      **{k:r[k] for k in ("iteration", *METRICS)}))
            for lo, hi in PHASES:
                val = [r for r in run["val"] if lo <= r["iteration"] <= hi]
                train = [r for r in run["train"] if lo <= int(r["iteration"]) <= hi]
                phase = dict(evaluations=len(val), training_log_points=len(train), validation={}, training={},
                             full_surface_evaluations=sum(r["surface_metric_cases"] == 20 for r in val))
                for key in METRICS:
                    phase["validation"][key] = dict(statistics(r[key] for r in val),
                        mean_delta_baseline=st.mean(r[key] - base_val[r["iteration"]][key] for r in val),
                        better_than_baseline=sum((r[key] - base_val[r["iteration"]][key]) * (1 if key in ("dice","jc") else -1) > 0 for r in val))
                for key in ("loss_supervised", "loss_consistency", "loss_boundary", "boundary_weight", "gradient_norm"):
                    if key in train[0]:
                        phase["training"][key] = statistics(float(r[key]) for r in train)
                audit["phases"][f"{lo}_{hi}"] = phase
                phase_rows.append(dict(dataset=dataset, condition=condition, lo=lo, hi=hi, evaluations=len(val),
                    **{f"mean_{key}": st.mean(r[key] for r in val) for key in METRICS},
                    **{f"delta_{key}": phase["validation"][key]["mean_delta_baseline"] for key in METRICS},
                    **{f"better_{key}": phase["validation"][key]["better_than_baseline"] for key in METRICS}))
                training_rows.append(dict(dataset=dataset, condition=condition, lo=lo, hi=hi,
                                          **{f"mean_{k}": v["mean"] for k,v in phase["training"].items()}))
            mature = [r for r in run["val"] if r["iteration"] >= 4000]
            audit["minimum_after4000"] = min(mature, key=lambda r:r["dice"])
            audit["drops_below_60pct_after4000"] = sum(r["dice"] < .6 for r in mature)
            if condition in ("e4", "d3"):
                frozen = js(path / "source_manifest.json")
                audit["snapshot_hashes_valid"] = all(digest(path / key) == h for key,h in frozen.items())
                assert audit["snapshot_hashes_valid"]
                assert [p["iteration"] for p in run["probes"]] == [1, *range(200,15001,200)]
                assert len(run["readout"]) == 72
                audit["probes_mature"] = summarize_probes(run["probes"], 4000, 15000)
                audit["probes_late"] = summarize_probes(run["probes"], 12001, 15000)
                for p in run["probes"]:
                    assert len(p["interventions"]) == 10
                    for r in p["interventions"]:
                        probe_rows.append(dict(dataset=dataset, condition=condition, iteration=p["iteration"],
                            group=r["group"], epsilon=r["epsilon"], predicted_delta=r["predicted_boundary_delta"],
                            removed_norm=r["removed_feature_norm"],
                            **{f"delta_{k}":v for k,v in r["actual_delta"].items()}))
                for r in run["readout"]:
                    readout_rows.append(dict(dataset=dataset, condition=condition,
                        **{k:r[k] for k in ("iteration", "loss", "precision", "recall", "f1")},
                        f1_valid_cases=sum(c["f1"] is not None for c in r["per_case"]),
                        empty_target_cases=sum(c["positive_voxels"] == 0 for c in r["per_case"])))
                audit["readout_at_main_best"] = {k:v for k,v in next(r for r in run["readout"] if r["iteration"] == best["iteration"]).items() if k != "per_case"}
                audit["readout_final"] = {k:v for k,v in run["readout"][-1].items() if k != "per_case"}
                audit["readout_final_f1_valid_cases"] = sum(c["f1"] is not None for c in run["readout"][-1]["per_case"])
                audit["readout_phases"] = {f"{lo}_{hi}": {key: statistics(r[key] for r in run["readout"] if lo <= r["iteration"] <= hi)
                                           for key in ("loss", "precision", "recall", "f1")} for lo,hi in PHASES}
                common = sorted(set(int(r["iteration"]) for r in run["train"]) & set(int(r["iteration"]) for r in base["train"]))
                if condition == "d3":
                    lookup = {int(r["iteration"]): r for r in run["train"]}
                    reference = {int(r["iteration"]): r for r in base["train"]}
                    steps = [s for s in common if s <= 2000]
                    audit["warmup_parity"] = {key:max(abs(float(lookup[s][key])-float(reference[s][key])) for s in steps)
                                              for key in ("loss_supervised", "loss_consistency", "gradient_norm")}
            dataset_report[condition] = audit
        # Shared batch-augmentation trace: target mass can only be compared within target families.
        reference = {int(r["iteration"]):r for r in runs["dual03"]["train"]}
        for condition in ("e4", "d3"):
            current = {int(r["iteration"]):r for r in runs[condition]["train"]}
            dataset_report[condition]["empty_foreground_trace_max_difference"] = max(
                abs(float(current[s]["empty_foreground_fraction"])-float(reference[s]["empty_foreground_fraction"])) for s in current)
            if condition == "d3":
                dataset_report[condition]["boundary_band_trace_max_difference"] = max(
                    abs(float(current[s]["boundary_band_fraction"])-float(reference[s]["boundary_band_fraction"])) for s in current)
                dataset_report[condition]["half_target_mass_max_difference"] = max(
                    abs(float(current[s]["boundary_positive_mass"])-float(reference[s]["boundary_half_positive_mass"])) for s in current)
        report["datasets"][dataset] = dataset_report
    save_csv("metrics.csv", metric_rows)
    save_csv("case_deltas.csv", case_rows)
    save_csv("validation_curves.csv", curve_rows)
    save_csv("validation_phases.csv", phase_rows)
    save_csv("training_phases.csv", training_rows)
    save_csv("intervention_deltas.csv", probe_rows)
    save_csv("readout_curves.csv", readout_rows)
    (OUT / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    plots(loaded)
    print(json.dumps(dict(status="PASS", output=str(OUT), primary_metrics=[r for r in metric_rows if r["condition"] in ("baseline","e4","d3")]), indent=2))


def plots(loaded):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(baseline="#333333", e4="#d07a16", d3="#1668b4", dual03="#959595")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, dataset in enumerate(DATASETS):
        for col, key in enumerate(("dice", "hd95", "asd")):
            ax = axes[row, col]
            for condition in ("baseline", "e4", "d3"):
                values = loaded[dataset][condition]["val"]
                ax.plot([r["iteration"] for r in values], [r[key]*(100 if key=="dice" else 1) for r in values],
                        color=colors[condition], label=condition, linewidth=1.3)
                if key != "dice":
                    partial=[r for r in values if r["surface_metric_cases"]<20]
                    if partial:
                        ax.scatter([r["iteration"] for r in partial],[r[key] for r in partial],
                                   marker="x",s=25,color=colors[condition],label=f"{condition}: 19 surface cases")
            ax.set_title(f"{dataset}: {key}")
            ax.set_xlabel("Training step")
            ax.set_ylabel("%" if key=="dice" else "voxel")
            ax.grid(alpha=.2)
            ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"validation_curves.{suffix}", dpi=180)
    for ax in axes.flat:
        ax.set_xlim(12000,15000)
        lines = [line.get_ydata()[np.array(line.get_xdata())>12000] for line in ax.lines]
        ymin, ymax = min(min(v) for v in lines), max(max(v) for v in lines)
        pad = max((ymax-ymin)*.1, .02)
        ax.set_ylim(ymin-pad,ymax+pad)
    fig.savefig(OUT / "late_validation_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,8))
    for row,dataset in enumerate(DATASETS):
        probes=[p for p in loaded[dataset]["d3"]["probes"] if p["iteration"]>=4000]
        for group,color in (("top","#d07a16"),("bottom","#1668b4"),("random","#888888")):
            values=[r for p in probes for r in p["interventions"] if r["epsilon"]==.1 and
                    (r["group"].startswith("random") if group=="random" else r["group"]==group)]
            axes[row,0].scatter([r["predicted_boundary_delta"] for r in values],
                                [r["actual_delta"]["boundary_loss"] for r in values],s=12,alpha=.55,color=color,label=group)
            axes[row,1].scatter([r["actual_delta"]["boundary_loss"] for r in values],
                                [r["actual_delta"]["final_boundary_ce"] for r in values],s=12,alpha=.55,color=color,label=group)
        ax=axes[row,0]
        lo,hi=ax.get_xlim();ax.plot([lo,hi],[lo,hi],color="black",linestyle="--",linewidth=.8)
        ax.set_title(f"{dataset}: D3 local finite intervention")
        ax.set_xlabel("Predicted boundary-loss change")
        ax.set_ylabel("Actual boundary-loss change")
        axes[row,1].set_title(f"{dataset}: D3 head vs final segmentation")
        axes[row,1].set_xlabel("Auxiliary boundary-loss change")
        axes[row,1].set_ylabel("Final segmentation boundary-CE change")
        for ax in axes[row]:
            ax.axhline(0,color="black",linewidth=.5);ax.axvline(0,color="black",linewidth=.5)
            ax.grid(alpha=.2);ax.legend()
    fig.tight_layout()
    for suffix in ("png","pdf"):
        fig.savefig(OUT / f"D3_intervention_calibration.{suffix}", dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,dataset in enumerate(DATASETS):
        for condition in ("e4","d3"):
            values=loaded[dataset][condition]["readout"]
            for col,key in enumerate(("f1","loss")):
                ax=axes[row,col]
                ax.plot([r["iteration"] for r in values],[r[key] for r in values],label=condition,color=colors[condition])
                ax.set_title(f"{dataset}: fixed-patch readout {key}")
                ax.set_xlabel("Training step");ax.grid(alpha=.2);ax.legend()
    fig.suptitle("E4 / D3 targets differ: interpret within-head learning trajectories")
    fig.tight_layout()
    fig.savefig(OUT / "readout_learning.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
