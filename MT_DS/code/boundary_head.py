"""Binary boundary/nonboundary BCE+Dice on two independent prediction heads."""
import torch
import torch.nn.functional as F


def independent_boundary_loss(outputs, encoded_band, eps=1e-6):
    if len(outputs) != 2:
        raise ValueError("Expected half/full boundary logits")
    target_full = (encoded_band[:, None] > 0).float()
    valid = target_full.flatten(1).sum(1) > 0
    terms, stats = [], {}
    for logits, name in zip(outputs, ("half", "full")):
        if logits.ndim != 5 or logits.shape[:2] != target_full.shape[:2]:
            raise ValueError("Expected Bx1xDxHxW boundary predictions")
        target = F.adaptive_avg_pool3d(target_full, logits.shape[2:])
        if valid.any():
            active_logits, active_target = logits[valid], target[valid]
            positive = active_target.sum()
            negative = active_target.numel() - positive
            pos_weight = (negative / positive.clamp_min(eps)).clamp(1, 100).detach()
            bce = F.binary_cross_entropy_with_logits(active_logits, active_target,
                                                     pos_weight=pos_weight)
            probability = active_logits.sigmoid()
            intersection = (probability * active_target).flatten(1).sum(1)
            denominator = probability.flatten(1).sum(1) + active_target.flatten(1).sum(1)
            dice = (1 - (2 * intersection + eps) / (denominator + eps)).mean()
            term = bce + dice
        else:
            term = logits.sum() * 0
            bce, dice = term, term
            positive, pos_weight = logits.new_tensor(0.), logits.new_tensor(1.)
        terms.append(term)
        stats.update({f"boundary_{name}": term.detach().item(),
                      f"boundary_{name}_bce": bce.detach().item(),
                      f"boundary_{name}_dice": dice.detach().item(),
                      f"boundary_{name}_valid_samples": int(valid.sum()),
                      f"boundary_{name}_positive_mass": positive.detach().item(),
                      f"boundary_{name}_pos_weight": pos_weight.detach().item()})
    return torch.stack(terms).mean(), stats
