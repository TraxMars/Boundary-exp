"""Sliding-window inference using the final full-resolution segmentation head."""
import itertools
import math

import numpy as np
import torch
from medpy import metric

from data import SPECS, case_path, load_case, read_ids


@torch.no_grad()
def predict_volume(model, image, patch_size, stride, device):
    padding = []
    for size, patch in zip(image.shape, patch_size):
        missing = max(patch - size, 0)
        padding.append((missing // 2, missing - missing // 2))
    padded = np.pad(image, padding)
    scores = np.zeros(padded.shape, dtype=np.float32)
    counts = np.zeros(padded.shape, dtype=np.float32)
    starts = [[min(step * delta, size - patch)
               for step in range(math.ceil((size - patch) / delta) + 1)]
              for size, patch, delta in zip(padded.shape, patch_size, stride)]
    for position in itertools.product(*starts):
        crop = tuple(slice(start, start + patch) for start, patch in zip(position, patch_size))
        inputs = torch.from_numpy(np.ascontiguousarray(padded[crop][None, None])).to(device)
        foreground = model(inputs).softmax(dim=1)[0, 1].cpu().numpy()
        scores[crop] += foreground
        counts[crop] += 1
    prediction = scores / counts > 0.5
    original = tuple(slice(left, left + size)
                     for (left, _), size in zip(padding, image.shape))
    return prediction[original]


def evaluate(model, root, dataset, patch_size, device, split=None, case_limit=None):
    split = split or SPECS[dataset]["validation"]
    ids = read_ids(root, split)
    if case_limit is not None:
        ids = ids[:case_limit]
    was_training = model.training
    model.eval()
    rows = []
    try:
        for case in ids:
            image, label = load_case(case_path(root, dataset, case))
            predicted = predict_volume(model, image, patch_size, SPECS[dataset]["stride"], device)
            truth = label.astype(bool)
            row = dict(case=case)
            if predicted.any() and truth.any():
                row.update(dice=float(metric.binary.dc(predicted, truth)),
                           jc=float(metric.binary.jc(predicted, truth)),
                           hd95=float(metric.binary.hd95(predicted, truth)),
                           asd=float(metric.binary.asd(predicted, truth)))
            else:
                both_empty = not predicted.any() and not truth.any()
                row.update(dice=float(both_empty), jc=float(both_empty), hd95=None, asd=None)
            rows.append(row)
    finally:
        model.train(was_training)
    summary = dict(split=split, cases=len(rows), per_case=rows,
                   surface_metric_cases=sum(row["hd95"] is not None for row in rows))
    for key in ("dice", "jc", "hd95", "asd"):
        values = [row[key] for row in rows if row[key] is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary
