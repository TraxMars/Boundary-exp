"""Historical E4 targets and native-scale D3 boundary readout objectives."""
import torch
import torch.nn.functional as F


def morphology_boundary(labels, radius=1):
    x = labels.float().unsqueeze(1)
    kernel = 2 * radius + 1
    maximum = F.max_pool3d(x, kernel, stride=1, padding=radius)
    minimum = -F.max_pool3d(-x, kernel, stride=1, padding=radius)
    return (maximum != minimum).float()


def probe_boundary_loss(logits, target_full, variant, eps=1e-6):
    target = (target_full.float() if variant == "e4" else
              F.adaptive_avg_pool3d(target_full.float(), logits.shape[2:]))
    valid = (torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
             if variant == "e4" else target_full.flatten(1).sum(1) > 0)
    if valid.any():
        active_logits, active_target = logits[valid], target[valid]
        positive = active_target.sum()
        negative = active_target.numel() - positive
        denominator = positive + eps if variant == "e4" else positive.clamp_min(eps)
        pos_weight = (negative / denominator).clamp(1, 100).detach()
        bce = F.binary_cross_entropy_with_logits(active_logits, active_target, pos_weight=pos_weight)
        probability = active_logits.sigmoid()
        intersection = (probability * active_target).flatten(1).sum(1)
        mass = probability.flatten(1).sum(1) + active_target.flatten(1).sum(1)
        ratio = (2 * intersection + eps) / (mass + eps)
        dice = 1 - ratio.mean() if variant == "e4" else (1 - ratio).mean()
        loss = bce + dice
    else:
        loss = logits.sum() * 0
        bce = dice = loss
        positive, pos_weight = logits.new_tensor(0.), logits.new_tensor(1.)
    return loss, dict(boundary_bce=bce.detach().item(), boundary_dice=dice.detach().item(),
                      boundary_valid_samples=int(valid.sum()),
                      boundary_positive_mass=positive.detach().item(),
                      boundary_pos_weight=pos_weight.detach().item())


def final_boundary_ce(outputs, labels, target_full):
    ce = F.cross_entropy(outputs[-1], labels.long(), reduction="none")
    mask = target_full[:, 0]
    mass = mask.flatten(1).sum(1)
    per_sample = (ce * mask).flatten(1).sum(1) / mass.clamp_min(1)
    valid = mass > 0
    return (per_sample * valid).sum() / valid.sum().clamp_min(1)
