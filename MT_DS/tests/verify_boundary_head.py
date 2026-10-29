"""Independent numerical, gradient, and zero-weight equivalence checks."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from boundary import gradient_probe, make_band
from boundary_head import independent_boundary_loss
from networks.vnet_ds import VNetDS
from networks.vnet_boundary_head import VNetBoundaryHead
from objectives import consistency_loss, supervised_loss, update_ema


def hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / "code").rglob("*.py"))}


def main():
    torch.set_num_threads(2)
    passed = []
    band = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    band[0, 1, 1, 1], band[0, 2, 2, 2] = 1, 2
    logits = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    value, stats = independent_boundary_loss((logits, logits), band)
    # With two positives, beta=31: positive and negative BCE mass both equal 62.
    expected_bce = torch.log(torch.tensor(2.)) * 124 / 64
    expected_dice = 1 - (2 + 1e-6) / (32 + 2 + 1e-6)
    torch.testing.assert_close(value, expected_bce + expected_dice)
    value.backward()
    assert logits.grad[0, 0, 1, 1, 1] < 0 and logits.grad[0, 0, 2, 2, 2] < 0
    assert logits.grad[0, 0, 0, 0, 0] > 0
    swapped = torch.where(band == 1, 2, torch.where(band == 2, 1, 0))
    alternate, _ = independent_boundary_loss((logits.detach(), logits.detach()), swapped)
    torch.testing.assert_close(value.detach(), alternate, rtol=0, atol=0)
    passed.append("analytic_BCE_Dice_and_both_band_sides_are_boundary_positives")

    half = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
    _, stats = independent_boundary_loss((half, logits), band)
    assert stats["boundary_half_positive_mass"] == 2 / 8
    assert stats["boundary_full_positive_mass"] == 2
    joined_band = torch.cat((band, torch.zeros_like(band)))
    mixed, _ = independent_boundary_loss((torch.cat((half, half)), torch.cat((logits, logits))), joined_band)
    single, _ = independent_boundary_loss((half, logits), band)
    torch.testing.assert_close(single, mixed)
    empty, info = independent_boundary_loss((half, logits), torch.zeros_like(band))
    assert empty.item() == 0 and info["boundary_full_valid_samples"] == 0
    empty.backward()
    passed.append("fractional_half_target_mass_empty_sample_exclusion_and_finite_empty_loss")

    torch.manual_seed(42)
    ordinary = VNetDS(1, 2, normalization="batchnorm", has_dropout=True).train()
    reference_rng = torch.get_rng_state()
    torch.manual_seed(42)
    added = VNetBoundaryHead(1, 2, normalization="batchnorm", has_dropout=True,
                            boundary_head_seed=1000045).train()
    assert torch.equal(reference_rng, torch.get_rng_state())
    for key, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, added.state_dict()[key], rtol=0, atol=0)
    passed.append("identical_backbone_initialization_and_preserved_training_RNG")

    teacher_a, teacher_b = copy.deepcopy(ordinary).requires_grad_(False), copy.deepcopy(added).requires_grad_(False)
    optimizer_a = torch.optim.SGD(ordinary.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    optimizer_b = torch.optim.SGD(added.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    labeled, unlabeled = torch.randn(2, 1, 32, 32, 32), torch.randn(2, 1, 32, 32, 32)
    labels = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    labels[:, 8:24, 8:24, 8:24] = 1
    bands = torch.stack([torch.from_numpy(make_band(label.numpy(), 2)) for label in labels])
    rng = torch.get_rng_state()
    out_a = ordinary(labeled, deepsup=True)
    un_a = ordinary(unlabeled, deepsup=True)
    with torch.no_grad():
        teach_a = teacher_a(unlabeled, deepsup=True)
    sup_a, _, _ = supervised_loss(out_a, labels)
    mt_a = consistency_loss(un_a, teach_a)
    (sup_a + .3 * mt_a).backward()
    norm_a = torch.nn.utils.clip_grad_norm_(ordinary.parameters(), float("inf"))
    optimizer_a.step()
    update_ema(ordinary, teacher_a, .99, 0)
    after_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    out_b, boundary_outputs = added(labeled, deepsup=True, return_boundary=True)
    un_b = added(unlabeled, deepsup=True)
    with torch.no_grad():
        teach_b = teacher_b(unlabeled, deepsup=True)
    assert boundary_outputs[0].shape == (2, 1, 16, 16, 16)
    assert boundary_outputs[1].shape == (2, 1, 32, 32, 32)
    for a, b in zip(out_a, out_b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sup_b, _, _ = supervised_loss(out_b, labels)
    mt_b = consistency_loss(un_b, teach_b)
    boundary, _ = independent_boundary_loss(boundary_outputs, bands)
    watched = [("half_upsampling", added.block_seven_up.conv[0].weight, 1),
               ("final_conv", added.block_nine.conv[0].weight, 0)]
    probe = gradient_probe(sup_b, boundary, watched, 0.)
    assert probe["boundary_norm"] > 0
    assert all(p.grad is None for p in added.parameters())
    assert torch.equal(after_rng, torch.get_rng_state())
    (sup_b + .3 * mt_b).backward()
    norm_b = torch.nn.utils.clip_grad_norm_(added.parameters(), float("inf"))
    torch.testing.assert_close(norm_a, norm_b, rtol=0, atol=0)
    assert all(p.grad is None for name, p in added.named_parameters() if name.startswith("boundary_"))
    optimizer_b.step()
    update_ema(added, teacher_b, .99, 0)
    for key, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, added.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(teacher_a.state_dict()[key], teacher_b.state_dict()[key], rtol=0, atol=0)
    assert all(p.grad is None for p in teacher_b.parameters())
    assert not added.out_conv._forward_pre_hooks and not added.out_conv_2._forward_pre_hooks
    passed.append("zero_weight_preserves_all_four_logits_BN_Dropout_SGD_EMA_and_teacher_isolation")

    optimizer_b.zero_grad(set_to_none=True)
    _, edges = added(labeled, deepsup=True, return_boundary=True)
    loss, _ = independent_boundary_loss(edges, bands)
    loss.backward()
    assert added.boundary_half[0].weight.grad.norm() > 0
    assert added.boundary_full[0].weight.grad.norm() > 0
    assert added.block_one.conv[0].weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in added.parameters() if p.grad is not None)
    assert all(p.grad is None for p in teacher_b.parameters())
    passed.append("active_boundary_only_gradient_reaches_both_auxiliary_heads_and_encoder")

    report = PROJECT / "verification_runs/independent_boundary_head_v1/unit_results.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(dict(status="PASS", checks=passed, source_hashes=hashes()), indent=2) + "\n")
    print(json.dumps(dict(status="PASS", checks=passed, report=str(report)), indent=2))


if __name__ == "__main__":
    main()
