"""Read historical online probe diagnostics; never load model checkpoints."""
import csv
import hashlib
import json
import statistics
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
OLD = PROJECT.parents[1] / "MT+DuCiSC+BGCE" / "MT-DuCiSC"
RUNS = {
    "LA_zero": "LA_BoundaryProbeW000_8_labeled_0.01_0.99_seed1337",
    "LA_probe": "LA_BoundaryProbeW003_8_labeled_0.01_0.99_seed1337",
    "Pancreas_probe": "Pancreas_BoundaryProbeW003_6_labeled_0.01_0.99_seed1337",
}


def main():
    result = {"scope": "saved training-time records only", "runs": {}}
    for label, name in RUNS.items():
        directory = OLD / "results" / name / "teacher_0.3_none"
        source = directory / "cbcd_occlusion.csv"
        with source.open() as stream:
            rows = list(csv.DictReader(stream))
        selected = [r for r in rows if 4000 < int(r["iteration"]) <= 15000]
        result["runs"][label] = {
            "directory": str(directory),
            "csv_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "all_rows": len(rows),
            "analysis_interval": "4000 < iteration <= 15000",
            "n": len(selected),
            "median": {
                key: statistics.median(float(r[key]) for r in selected)
                for key in ("top_delta", "bottom_delta", "random_delta")
            },
            "top_gt_random": sum(float(r["top_delta"]) > float(r["random_delta"])
                                 for r in selected),
            "top_positive": sum(float(r["top_delta"]) > 0 for r in selected),
        }
    result["limits"] = [
        "Repeated diagnostic batches from one seed, not independent replications.",
        "Only auxiliary probe is recomputed; segmentation decoder is not replayed.",
        "Scores use train-mode pre-update features; occlusion uses eval-mode probe after SGD.",
        "Absolute elementwise gradient-times-feature scores do not encode signed utility.",
        "Zero-weight probe is untrained; its edge accuracy is not a fair readout control.",
    ]
    output = Path(__file__).with_name("comparison.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
