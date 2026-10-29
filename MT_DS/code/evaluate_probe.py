"""Fixed center-patch boundary readout monitoring; never used for selection."""
import numpy as np
import torch
import torch.nn.functional as F

from boundary import make_band
from data import SPECS, case_path, load_case, read_ids
from online_probe import diagnostic_state
from probe_objectives import morphology_boundary, probe_boundary_loss


@torch.no_grad()
def evaluate_readout(model, root, dataset, patch_size, device, split=None):
    split = split or SPECS[dataset]["validation"]
    rows = []
    with diagnostic_state(model):
        for case in read_ids(root, split):
            image, label = load_case(case_path(root, dataset, case))
            band = make_band(label, 2) if model.variant == "d3" else None
            pads = [(max(p - s, 0) // 2, max(p - s, 0) - max(p - s, 0) // 2)
                    for p, s in zip(patch_size, label.shape)]
            image, label = np.pad(image, pads), np.pad(label, pads)
            if band is not None:
                band = np.pad(band, pads)
            crop = tuple(slice((s - p) // 2, (s - p) // 2 + p)
                         for p, s in zip(patch_size, label.shape))
            inputs = torch.from_numpy(np.ascontiguousarray(image[crop][None, None])).to(device)
            labels = torch.from_numpy(np.ascontiguousarray(label[crop][None])).to(device)
            if band is None:
                target = morphology_boundary(labels)
            else:
                target = torch.from_numpy(np.ascontiguousarray((band[crop] > 0)[None, None])).float().to(device)
            _, logits, _ = model(inputs, deepsup=True, return_boundary=True)
            loss, stats = probe_boundary_loss(logits, target, model.variant)
            full_logits = F.interpolate(logits, size=patch_size, mode="trilinear", align_corners=False)
            predicted, truth = full_logits.sigmoid() > .5, target > 0
            tp = int((predicted & truth).sum())
            fp, fn = int((predicted & ~truth).sum()), int((~predicted & truth).sum())
            precision = tp / (tp + fp) if tp + fp else 0.
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            rows.append(dict(case=case, loss=loss.item(), precision=precision, recall=recall,
                             f1=f1, tp=tp, fp=fp, fn=fn, positive_voxels=int(truth.sum()), **stats))
    result = dict(split=split, scope="fixed_center_patch_per_case_not_whole_volume_surface_metric",
                  threshold=.5, prediction_resize="trilinear_logits_then_sigmoid",
                  used_for_selection=False, cases=len(rows), per_case=rows)
    for key in ("loss", "precision", "recall", "f1"):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result
