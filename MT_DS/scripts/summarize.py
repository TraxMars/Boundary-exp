"""Collect all six test results and compare MT_DS with its preserved parent."""
import argparse
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]
RUNS = (("LA", 10, 8), ("LA", 20, 16), ("Pancreas", 10, 6),
        ("Pancreas", 20, 12), ("BraTS19", 10, 25), ("BraTS19", 20, 50))


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def collect(project, seed=42):
    rows = []
    for dataset, percent, labels in RUNS:
        root = project / "Results" / f"seed_{seed}" / f"{dataset}_{labels}l"
        status = read_json(root / "run_status.json")
        best = read_json(root / "best_metrics.json")
        test = read_json(root / "test_metrics.json")
        rows.append(dict(
            version=project.name, dataset=dataset, label_percent=percent, labelnum=labels,
            seed=seed, state=status.get("state", "pending"), iteration=status.get("iteration", 0),
            best_iteration=best.get("iteration"), selection_split=best.get("split"),
            best_selection_dice=best.get("dice"),
            test_dice=test.get("dice"), test_jc=test.get("jc"),
            test_hd95=test.get("hd95"), test_asd=test.get("asd"),
            test_cases=test.get("cases"), surface_metric_cases=test.get("surface_metric_cases"),
            run_directory=str(root.resolve()),
            best_metrics_path=str((root / "best_metrics.json").resolve()),
            test_metrics_path=str((root / "test_metrics.json").resolve())))
    return rows


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def display(value, percent=False):
    return "—" if value is None else f"{value * (100 if percent else 1):.2f}"


def save_summary(project, rows):
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    notes = ["Main results use test_metrics.json, never validation scores as test scores.",
             "LA/Pancreas use test.list for online model selection, as in the parent recipe.",
             "BraTS19 selects on val.list and reports the selected model on test.list.",
             "Dice/JC are fractions in JSON/CSV and percentages in Markdown; distances are voxel units."]
    directory = project / "summary"
    atomic_json(directory / "results.json", dict(updated_at=now, notes=notes, runs=rows))
    atomic_text(directory / "results.csv", csv_text(rows))
    lines = [f"# {project.name}：六组结果", "", f"更新时间：{now}", "",
             "下表为最佳模型的最终测试结果。Dice / JC 为百分数；HD95 / ASD 为体素单位。", "",
             "| 数据集 | 标注比例 | 标注例数 | 状态 | 最佳步数 | Test Dice ↑ | Test JC ↑ | HD95 ↓ | ASD ↓ |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['label_percent']}% | {row['labelnum']} | "
                     f"{row['state']} | {row['best_iteration'] or '—'} | "
                     f"{display(row['test_dice'], True)} | {display(row['test_jc'], True)} | "
                     f"{display(row['test_hd95'])} | {display(row['test_asd'])} |")
    lines += ["", "LA / Pancreas 沿用 test.list 在线选模；BraTS19 在 val.list 选模后独立测试。", "",
              "BraTS19 的在线验证 Dice（与最终测试分开）：", ""]
    for row in rows:
        if row["dataset"] == "BraTS19":
            lines.append(f"- {row['label_percent']}%：{display(row['best_selection_dice'], True)}%，"
                         f"最佳步数 {row['best_iteration'] or '—'}。")
    atomic_text(directory / "results.md", "\n".join(lines) + "\n")


def refresh(project=PROJECT, baseline=None, seed=42):
    baseline = baseline or project.parent / "MT_DS_LossNorm"
    current = collect(project, seed)
    previous = collect(baseline, seed)
    save_summary(project, current)
    comparisons = []
    for before, after in zip(previous, current):
        delta = (after["test_dice"] - before["test_dice"]) * 100 if (
            after["test_dice"] is not None and before["test_dice"] is not None) else None
        comparisons.append(dict(dataset=after["dataset"], label_percent=after["label_percent"],
                                labelnum=after["labelnum"], seed=seed, state=after["state"],
                                lossnorm_test_dice=before["test_dice"],
                                ds_only_test_dice=after["test_dice"],
                                delta_dice_percentage_points=delta,
                                lossnorm_test_hd95=before["test_hd95"],
                                ds_only_test_hd95=after["test_hd95"]))
    atomic_json(project / "summary/comparison.json", dict(baseline=previous, current=current,
                                                         comparison=comparisons))
    atomic_text(project / "summary/comparison.csv", csv_text(comparisons))
    lines = ["# 损失归一化消融对比", "", "使用最佳模型的最终 Test Dice，单位为百分数。", "",
             "| 数据集 | 标注比例 | MT+DS+LossNorm | MT+DS | 差值（百分点） | MT+DS 状态 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in comparisons:
        delta = row["delta_dice_percentage_points"]
        delta_text = "—" if delta is None else f"{delta:+.2f}"
        lines.append(f"| {row['dataset']} | {row['label_percent']}% | "
                     f"{display(row['lossnorm_test_dice'], True)} | "
                     f"{display(row['ds_only_test_dice'], True)} | {delta_text} | {row['state']} |")
    atomic_text(project / "summary/comparison.md", "\n".join(lines) + "\n")
    return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--baseline", type=Path, default=PROJECT.parent / "MT_DS_LossNorm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    save_summary(args.baseline, collect(args.baseline, args.seed))
    refresh(args.project, args.baseline, args.seed)
    print(args.project / "summary/comparison.md")
