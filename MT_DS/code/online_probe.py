"""Read-only, pre-update channel interventions with a frozen decoder state."""
from contextlib import contextmanager
import math

import torch

from objectives import supervised_loss
from probe_objectives import final_boundary_ce, probe_boundary_loss


@contextmanager
def diagnostic_state(model):
    modes = [(module, module.training) for module in model.modules()]
    device = next(model.parameters()).device
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        for module, mode in modes:
            module.training = mode


def _metrics(outputs, logits, labels, target, variant):
    boundary, _ = probe_boundary_loss(logits, target, variant)
    region, _, _ = supervised_loss(outputs, labels)
    prediction = outputs[-1].argmax(1)
    truth = labels == 1
    intersection = ((prediction == 1) & truth).flatten(1).sum(1)
    denominator = (prediction == 1).flatten(1).sum(1) + truth.flatten(1).sum(1)
    dice = torch.where(denominator > 0, 2 * intersection.float() / denominator.clamp_min(1), 1.)
    return dict(boundary_loss=boundary.item(), region_loss=region.item(),
                final_boundary_ce=final_boundary_ce(outputs, labels, target).item(),
                patch_dice=dice.mean().item(), foreground_fraction=(prediction == 1).float().mean().item())


def _correlation(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = a.norm() * b.norm()
    return (a.dot(b) / denominator).item() if denominator > 1e-15 else None


def channel_interventions(model, features, labels, target, *, step, seed, fraction=.125,
                          random_repeats=3, epsilons=(.1, 1.)):
    features = tuple(f.detach() for f in features)
    index = 3 if model.variant == "e4" else 1
    channels = features[index].shape[1]
    k = max(1, min(channels, round(channels * fraction)))
    devices = features[index].device
    with diagnostic_state(model), torch.enable_grad():
        gate = torch.ones((features[0].shape[0], channels), device=devices, requires_grad=True)
        outputs, logits, observed = model.decode_probe(features, gate)
        boundary, _ = probe_boundary_loss(logits, target, model.variant)
        if model.variant == "e4":
            gradient = torch.autograd.grad(boundary, observed)[0]
            signed = -(gradient * observed).flatten(2).sum(2).detach()
            scores = (gradient * observed).abs().flatten(2).mean(2).detach()
        else:
            signed = -torch.autograd.grad(boundary, gate)[0].detach()
            scores = signed
        baseline = _metrics(outputs, logits, labels, target, model.variant)
        del outputs, logits, observed, boundary, gate
        generator = torch.Generator(device="cpu").manual_seed(seed * 10000019 + step)
        groups = [("top", scores.topk(k, dim=1).indices.cpu()),
                  ("bottom", scores.topk(k, dim=1, largest=False).indices.cpu())]
        for repeat in range(random_repeats):
            groups.append((f"random_{repeat}", torch.stack([
                torch.randperm(channels, generator=generator)[:k] for _ in range(scores.shape[0])
            ])))
        interventions = []
        with torch.no_grad():
            for epsilon in epsilons:
                for name, indices in groups:
                    changed = torch.ones_like(scores)
                    indices_device = indices.to(devices)
                    changed.scatter_(1, indices_device, 1 - epsilon)
                    outputs, logits, _ = model.decode_probe(features, changed)
                    measured = _metrics(outputs, logits, labels, target, model.variant)
                    predicted = epsilon * signed.gather(1, indices_device).sum().item()
                    removed = features[index] * (1 - changed)[..., None, None, None]
                    interventions.append(dict(
                        group=name, epsilon=epsilon, channels=indices.tolist(),
                        predicted_boundary_delta=predicted,
                        actual_delta={key: value - baseline[key] for key, value in measured.items()},
                        removed_feature_norm=removed.norm().item()))
                    del outputs, logits, removed
        small = [row for row in interventions if row["epsilon"] == min(epsilons)]
        predicted = [row["predicted_boundary_delta"] for row in small]
        actual = [row["actual_delta"]["boundary_loss"] for row in small]
        valid_pairs = [(p, a) for p, a in zip(predicted, actual) if abs(p) > 1e-8 and abs(a) > 1e-8]
        result = dict(
            iteration=step, variant=model.variant, k=k, channels=channels,
            score_type=("absolute_elementwise_gradient_times_E4_probe_feature" if model.variant == "e4"
                        else "signed_negative_D3_boundary_derivative_wrt_E2_skip_gate"),
            scope="cached_training_encoder_features_eval_decoder_pre_optimizer_update",
            score=scores.cpu().tolist(), signed_removal_score=signed.cpu().tolist(),
            baseline=baseline, interventions=interventions,
            small_step_group_pearson=_correlation(predicted, actual),
            small_step_group_sign_agreement=(sum(p * a > 0 for p, a in valid_pairs) / len(valid_pairs)
                                             if valid_pairs else None),
            informative_boundary_samples=int((target.flatten(1).sum(1) > 0).sum()),
            score_norm=scores.norm().item())
        if not math.isfinite(result["score_norm"]):
            raise FloatingPointError("Nonfinite channel scores")
        return result
