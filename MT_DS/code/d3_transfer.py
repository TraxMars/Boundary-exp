"""Boundary-prototype L-to-U residuals, signed donor ranking, and read-only checks."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from data import VolumeDataset
from online_probe import diagnostic_state, _metrics
from probe_objectives import morphology_boundary, probe_boundary_loss


class IdentifiedVolumeDataset(VolumeDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["case_id"] = self.ids[index]
        return sample


def interaction_strength(completed, maximum=.1, start=4000, ramp=1000):
    return maximum * min(1., max(0., (completed + 1 - start) / max(ramp, 1)))


def signed_donor_scores(model, features, target):
    """Same conditional eval-decoder gate derivative as the stage-1 diagnostic."""
    detached = tuple(f.detach() for f in features)
    with diagnostic_state(model), torch.enable_grad():
        gate = torch.ones(detached[1].shape[:2], device=detached[1].device, requires_grad=True)
        outputs, logits, _ = model.decode_probe(detached, gate)
        loss, _ = probe_boundary_loss(logits, target, "d3")
        score = -torch.autograd.grad(loss, gate)[0].detach()
        if not torch.isfinite(score).all():
            raise FloatingPointError("Nonfinite donor scores")
        return score


class BoundaryPrototypeTransfer(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv3d(channels, channels, 1, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Conv3d(channels, channels, 1, bias=False)

    def forward(self, donor_feature, donor_band, scores, receiver_feature, receiver_logits,
                *, mode, alpha, seed, step, fraction=.125):
        if mode not in ("random", "boundary"):
            raise ValueError(mode)
        donor_feature, scores = donor_feature.detach(), scores.detach()
        if donor_feature.shape[:2] != receiver_feature.shape[:2]:
            raise ValueError("Use fixed one-to-one pairing with equal labeled/unlabeled batch sizes")
        k = max(1, min(self.channels, round(self.channels * fraction)))
        inside = F.adaptive_avg_pool3d((donor_band[:,None] == 1).float(), donor_feature.shape[2:])
        outside = F.adaptive_avg_pool3d((donor_band[:,None] == 2).float(), donor_feature.shape[2:])
        masses = torch.stack((inside.flatten(1).sum(1), outside.flatten(1).sum(1)), 1)
        valid = (masses > 1e-6).all(1) & ((scores > 0).sum(1) >= k)
        if mode == "boundary":
            indices = scores.topk(k, dim=1).indices
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed * 10000019 + step)
            indices = torch.stack([torch.randperm(self.channels, generator=generator)[:k]
                                   for _ in range(scores.shape[0])]).to(scores.device)
        channel_mask = torch.zeros_like(scores).scatter_(1, indices, 1.)
        # Keep original channel coordinates; never align samples by top-k rank.
        prototypes = torch.stack([
            (donor_feature * mask).flatten(2).sum(2) / mask.flatten(2).sum(2).clamp_min(1e-6)
            for mask in (inside,outside)], 1)
        prototypes = (prototypes * channel_mask[:,None]).detach()
        with torch.no_grad():
            probability = receiver_logits.detach().softmax(1)
            hard = probability.argmax(1)
            pseudo_band = morphology_boundary(hard, radius=2)
            confidence = (2 * probability.max(1, keepdim=True).values - 1).clamp(0,1)
            spatial_mask = F.adaptive_avg_pool3d(pseudo_band * confidence, receiver_feature.shape[2:])
            spatial_mask = spatial_mask * valid[:,None,None,None,None]
        query = self.query(receiver_feature)
        query = F.normalize(query, dim=1, eps=1e-6)
        keys = F.normalize(self.key(prototypes), dim=2, eps=1e-6)
        attention = torch.einsum("bcxyz,bpc->bpxyz", query, keys).softmax(1)
        # Centered inner/outer values focus on their feature contrast.
        values = self.value(prototypes - prototypes.mean(1, keepdim=True))
        raw = self.output(torch.einsum("bpxyz,bpc->bcxyz", attention, values))
        raw = raw * channel_mask[...,None,None,None] * spatial_mask
        with torch.no_grad():
            reference_norm = (receiver_feature.detach() * spatial_mask).flatten(1).norm(dim=1) * math.sqrt(k/self.channels)
            raw_norm = raw.detach().flatten(1).norm(dim=1)
            scale = (alpha * reference_norm / raw_norm.clamp_min(1e-8)).clamp(max=1.)
        residual = raw * scale[:,None,None,None,None]
        actual_norm = residual.detach().flatten(1).norm(dim=1)
        ratio = actual_norm / reference_norm.clamp_min(1e-8)
        stats = dict(
            interaction_alpha=float(alpha), interaction_valid_donors=int(valid.sum()),
            interaction_active_samples=int(((actual_norm>0) & valid).sum()),
            interaction_mask_mean=float(spatial_mask.mean()),
            interaction_residual_ratio_mean=float(ratio.mean()),
            interaction_residual_ratio_max=float(ratio.max()),
            interaction_residual_norm=float(actual_norm.norm()), interaction_channels=k)
        route = dict(indices=indices.cpu().tolist(), valid_donors=valid.cpu().tolist(),
                     scores=scores.cpu().tolist(), selected_scores=scores.gather(1,indices).cpu().tolist(),
                     donor_side_masses=masses.cpu().tolist(), relative_residual_norm=ratio.cpu().tolist())
        return residual, stats, route


@torch.no_grad()
def labeled_injection_check(model, features, labels, band, scores, *, mode, alpha, seed, step, fraction):
    """GT evaluates receivers; the residual location uses a clean prediction only."""
    features = tuple(f.detach() for f in features)
    target = (band[:,None]>0).float()
    with diagnostic_state(model):
        outputs, logits, _ = model.decode_probe(features)
        baseline = _metrics(outputs, logits, labels, target, "d3")
        residual, stats, route = model.transfer(
            features[1].flip(0), band.flip(0), scores.flip(0), features[1], outputs[-1],
            mode=mode, alpha=alpha, seed=seed+170003, step=step, fraction=fraction)
        branch = list(features)
        branch[1] = branch[1] + residual
        outputs, logits, _ = model.decode_probe(branch)
        changed = _metrics(outputs, logits, labels, target, "d3")
        return dict(iteration=step, baseline=baseline,
                    actual_delta={k:changed[k]-baseline[k] for k in baseline},
                    **stats, route=route,
                    scope="other_labeled_sample_donor_prediction_only_receiver_mask_eval_decoder_no_updates")
