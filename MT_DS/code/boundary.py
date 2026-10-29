"""A1 boundary-band targets, balanced CE, and observational gradient probes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def make_band(label, width):
    """1 = foreground within width of background; 2 = background near foreground.

    Distances are between voxel centers in the complete supplied volume. No
    artificial background padding is added. Degenerate masks have no boundary.
    """
    from scipy.ndimage import distance_transform_edt
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Boundary width must be positive and finite")
    foreground = np.asarray(label, dtype=bool)
    band = np.zeros(foreground.shape, dtype=np.uint8)
    if foreground.any() and not foreground.all():
        band[foreground & (distance_transform_edt(foreground) <= width)] = 1
        band[~foreground & (distance_transform_edt(~foreground) <= width)] = 2
    return band


def prepare_cache(root, dataset, ids, width, cache_root):
    """Cache GT-derived bands for the explicitly labeled case IDs only."""
    from data import case_path, load_case
    signature = hashlib.sha256(json.dumps(
        dict(version=1, root=str(Path(root).resolve()), dataset=dataset,
             ids=list(ids), width=width), sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(cache_root) / f"{dataset}_{signature}"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {}
    for index, case in enumerate(ids):
        source = case_path(root, dataset, case)
        stat = source.stat()
        fingerprint = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        target = directory / f"{index:03d}.npy"
        old = previous.get("cases", {}).get(case, {})
        if not target.exists() or old.get("source_stat") != fingerprint:
            _, label = load_case(source)
            band = make_band(label, width)
            temporary = target.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, band, allow_pickle=False)
            temporary.replace(target)
            old = dict(source=str(source), source_stat=fingerprint,
                       label_sha256=hashlib.sha256(label.tobytes()).hexdigest(),
                       shape=list(label.shape), inner_voxels=int((band == 1).sum()),
                       outer_voxels=int((band == 2).sum()))
        records[case] = dict(old, file=target.name)
    payload = dict(version=1, dataset=dataset, width_voxel=width,
                   definition="full_volume_voxel_center_distance_no_added_padding",
                   cases=records)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(manifest_path)
    return directory


def boundary_weight(step, maximum, start, ramp):
    """step is the number of optimizer updates already completed."""
    return maximum * min(1.0, max(0.0, (step + 1 - start) / max(ramp, 1)))


def balanced_boundary_loss(outputs, encoded_band):
    """Average valid sample-side CEs, then average half/full-resolution heads.

    Area pooling supplies fractional inner/outer occupancy. A coarse cell may
    supervise both sides, with each side retaining its own class target and mass.
    This avoids nearest-neighbor disappearance of narrow bands.
    """
    inner = (encoded_band[:, None] == 1).float()
    outer = (encoded_band[:, None] == 2).float()
    terms, stats = [], {}
    for index, name in ((-2, "half"), (-1, "full")):
        logits = outputs[index]
        logp = logits.log_softmax(1)
        inside = F.adaptive_avg_pool3d(inner, logits.shape[2:])[:, 0]
        outside = F.adaptive_avg_pool3d(outer, logits.shape[2:])[:, 0]
        masses = torch.stack((inside.flatten(1).sum(1), outside.flatten(1).sum(1)), 1)
        numerators = torch.stack((
            -(inside * logp[:, 1]).flatten(1).sum(1),
            -(outside * logp[:, 0]).flatten(1).sum(1)), 1)
        valid = masses > 0
        side_losses = numerators / masses.clamp_min(1e-8)
        per_sample = (side_losses * valid).sum(1) / valid.sum(1).clamp_min(1)
        valid_samples = valid.any(1)
        term = (per_sample * valid_samples).sum() / valid_samples.sum().clamp_min(1)
        terms.append(term)
        stats[f"boundary_{name}"] = term.detach().item()
        stats[f"boundary_{name}_valid_samples"] = int(valid_samples.sum())
        stats[f"boundary_{name}_inner_mass"] = float(masses[:, 0].sum())
        stats[f"boundary_{name}_outer_mass"] = float(masses[:, 1].sum())
    return torch.stack(terms).mean(), stats


def gradient_probe(region_loss, boundary_loss, named_parameters, weight):
    """Observe existing graphs without backward(), RNG, forwards, or .grad writes."""
    parameters = [parameter for _, parameter, _ in named_parameters]
    region = torch.autograd.grad(region_loss, parameters, retain_graph=True)
    boundary = torch.autograd.grad(boundary_loss, parameters, retain_graph=True)
    rows = {}
    sums = [0.0, 0.0, 0.0]
    for (name, _, axis), gr, gb in zip(named_parameters, region, boundary):
        gr, gb = gr.detach(), gb.detach()
        rr, bb, rb = (gr.square().sum().item(), gb.square().sum().item(),
                      (gr * gb).sum().item())
        if not np.isfinite([rr, bb, rb]).all():
            raise FloatingPointError("Nonfinite boundary probe gradients")
        sums = [a + b for a, b in zip(sums, (rr, bb, rb))]
        nr, nb = rr ** 0.5, bb ** 0.5
        channel_r = gr.movedim(axis, 0).flatten(1)
        channel_b = gb.movedim(axis, 0).flatten(1)
        channel_norm_r = channel_r.norm(dim=1)
        channel_norm_b = channel_b.norm(dim=1)
        usable = (channel_norm_r > 1e-12) & (channel_norm_b > 1e-12)
        rows[name] = dict(region_norm=nr, boundary_norm=nb,
                          cosine=rb / (nr * nb) if nr * nb > 1e-20 else None,
                          weighted_ratio=weight * nb / nr if nr > 1e-12 else None,
                          channel_region_norm=channel_norm_r.cpu().tolist(),
                          channel_boundary_norm=channel_norm_b.cpu().tolist(),
                          channel_cosine=F.cosine_similarity(channel_r, channel_b, dim=1).cpu().tolist(),
                          channel_valid=usable.cpu().tolist(), output_channel_axis=axis)
    rr, bb, rb = sums
    nr, nb = rr ** 0.5, bb ** 0.5
    return dict(region_norm=nr, boundary_norm=nb,
                cosine=rb / (nr * nb) if nr * nb > 1e-20 else None,
                weighted_ratio=weight * nb / nr if nr > 1e-12 else None,
                parameters=rows)
