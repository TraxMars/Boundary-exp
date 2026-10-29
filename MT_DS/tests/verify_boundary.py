"""Numerical, data-isolation, and optimizer-equivalence checks for A1."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from boundary import balanced_boundary_loss, boundary_weight, gradient_probe, make_band
from data import VolumeDataset
from networks import VNetDS
from objectives import supervised_loss


def source_hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("code", "scripts", "tests")
            for p in sorted((PROJECT / directory).rglob("*.py"))}


def main():
    torch.set_num_threads(2)
    passed = []
    label = np.zeros((12, 12, 12), dtype=np.int64)
    label[3:9, 3:9, 3:9] = 1
    band = make_band(label, 2)
    assert band[3, 5, 5] == 1 and band[2, 5, 5] == 2 and band[0, 0, 0] == 0
    assert band[5, 5, 5] == 0
    assert not make_band(np.zeros_like(label), 2).any()
    assert not make_band(np.ones_like(label), 2).any()
    passed.append("full_volume_distance_bands_and_degenerate_masks")

    # Independently evaluate fine-grid expectation for a coarse (mixed-side) cell.
    encoded = torch.tensor([[[[1, 2], [0, 1]], [[2, 0], [1, 2]]]], dtype=torch.uint8)
    coarse = torch.tensor([[[[[0.3]]], [[[0.9]]]]], requires_grad=True)
    fine = torch.tensor([[[[[0.3, -0.2], [0.4, 0.2]], [[0.8, 0.1], [0.7, -0.4]]],
                          [[[0.9, 0.7], [-0.1, 0.5]], [[0.2, 0.6], [0.8, 0.3]]]]], requires_grad=True)
    actual, _ = balanced_boundary_loss([coarse, coarse, coarse, fine], encoded)
    expanded = F.interpolate(coarse, size=encoded.shape[1:], mode="nearest")
    expected_terms = []
    for logits in (expanded, fine):
        logp = logits.log_softmax(1)
        expected_terms.append((-logp[:, 1][encoded == 1].mean()
                               -logp[:, 0][encoded == 2].mean()) / 2)
    torch.testing.assert_close(actual, torch.stack(expected_terms).mean())
    actual.backward()
    assert torch.isfinite(coarse.grad).all() and coarse.grad.abs().sum() > 0
    assert torch.isfinite(fine.grad).all() and fine.grad.abs().sum() > 0
    empty, _ = balanced_boundary_loss([fine] * 4, torch.zeros_like(encoded))
    torch.testing.assert_close(empty, torch.zeros_like(empty))
    assert boundary_weight(1999, 0.1, 2000, 2000) == 0
    assert boundary_weight(2000, 0.1, 2000, 2000) == 0.1 / 2000
    assert boundary_weight(3999, 0.1, 2000, 2000) == 0.1
    passed.append("fractional_band_CE_matches_voxel_expectation_and_schedule")

    # Same RNG draws and exact images/labels, despite joint band augmentation.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "Pancreas_h5").mkdir()
        (root / "bands").mkdir()
        (root / "train.list").write_text("a\nb\n")
        full_label = np.zeros((40, 40, 40), dtype=np.int64)
        full_label[8:32, 8:32, 8:32] = 1
        full_band = make_band(full_label, 2)
        image = np.arange(40 ** 3, dtype=np.float32).reshape(full_label.shape)
        for case in ("a", "b"):
            with h5py.File(root / f"Pancreas_h5/{case}_norm.h5", "w") as handle:
                handle["image"], handle["label"] = image, full_label
        np.save(root / "bands/000.npy", full_band)
        ordinary = VolumeDataset(root, "Pancreas", (32, 32, 32))
        boundary = VolumeDataset(root, "Pancreas", (32, 32, 32), root / "bands", 1)
        np.random.seed(42)
        reference = ordinary[0]
        state = np.random.get_state()
        np.random.seed(42)
        added = boundary[0]
        next_state = np.random.get_state()
        assert np.array_equal(state[1], next_state[1]) and state[2:] == next_state[2:]
        assert torch.equal(reference["image"], added["image"])
        assert torch.equal(reference["label"], added["label"])
        start = tuple(int(v) for v in np.unravel_index(int(added["image"][0, 0, 0, 0]), image.shape))
        crop = tuple(slice(v, v + 32) for v in start)
        np.testing.assert_array_equal(added["boundary_band"].numpy(), full_band[crop])
        assert not boundary[1]["boundary_band"].any()
        # Exercise LA's rotation/flip and padding without introducing extra draws.
        la_dir = root / "2018LA_Seg_TrainingSet/a"
        la_dir.mkdir(parents=True)
        with h5py.File(la_dir / "mri_norm2.h5", "w") as handle:
            handle["image"], handle["label"] = image, full_label
        ordinary_la = VolumeDataset(root, "LA", (48, 48, 48))
        boundary_la = VolumeDataset(root, "LA", (48, 48, 48), root / "bands", 1)
        np.random.seed(7)
        reference = ordinary_la[0]
        state = np.random.get_state()
        np.random.seed(7)
        added = boundary_la[0]
        next_state = np.random.get_state()
        assert np.array_equal(state[1], next_state[1]) and state[2:] == next_state[2:]
        assert torch.equal(reference["image"], added["image"])
        assert torch.equal(reference["label"], added["label"])
        assert not (added["boundary_band"][added["label"] == 0] == 1).any()
        assert not (added["boundary_band"][added["label"] == 1] == 2).any()
    passed.append("LA_and_Pancreas_augmentation_RNG_and_unlabeled_band_isolation")

    # A probe/zero coefficient must leave the entire training update unchanged.
    torch.manual_seed(42)
    model_a = VNetDS(1, 2, normalization="batchnorm", has_dropout=True).train()
    model_b = copy.deepcopy(model_a)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.01, momentum=0.9)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.01, momentum=0.9)
    inputs = torch.randn(2, 1, 32, 32, 32)
    labels = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    labels[:, 8:24, 8:24, 8:24] = 1
    bands = torch.from_numpy(np.stack([make_band(v.numpy(), 2) for v in labels]))
    rng = torch.get_rng_state()
    loss_a, _, _ = supervised_loss(model_a(inputs, deepsup=True), labels)
    loss_a.backward()
    optimizer_a.step()
    reference_rng = torch.get_rng_state()
    torch.set_rng_state(rng)
    outputs_b = model_b(inputs, deepsup=True)
    region_b, _, _ = supervised_loss(outputs_b, labels)
    boundary_b, _ = balanced_boundary_loss(outputs_b, bands)
    probe_params = [("half", model_b.block_seven_up.conv[0].weight, 1),
                    ("full", model_b.block_nine.conv[0].weight, 0)]
    before = {name: tensor.clone() for name, tensor in model_b.state_dict().items()}
    probe = gradient_probe(region_b, boundary_b, probe_params, 0.0)
    assert probe["boundary_norm"] > 0 and probe["region_norm"] > 0
    assert all(p.grad is None for p in model_b.parameters())
    assert torch.equal(reference_rng, torch.get_rng_state())
    assert all(torch.equal(before[k], v) for k, v in model_b.state_dict().items())
    region_b.backward()
    optimizer_b.step()
    for key, value in model_a.state_dict().items():
        torch.testing.assert_close(value, model_b.state_dict()[key], rtol=0, atol=0)
    passed.append("zero_weight_probe_preserves_parameters_BN_Dropout_RNG_and_SGD_update")
    optimizer_b.zero_grad(set_to_none=True)
    outputs = model_b(inputs, deepsup=True)
    region, _, _ = supervised_loss(outputs, labels)
    added_loss, _ = balanced_boundary_loss(outputs, bands)
    (region + 0.1 * added_loss).backward()
    assert model_b.out_conv_2.weight.grad.norm() > 0 and model_b.out_conv.weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model_b.parameters() if p.grad is not None)
    passed.append("active_boundary_loss_reaches_both_heads_and_backbone")

    result = dict(status="PASS", checks=passed, source_hashes=source_hashes())
    report = PROJECT / "verification_runs/A1_boundary_v1/unit_results.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", checks=passed, report=str(report)), indent=2))


if __name__ == "__main__":
    main()
