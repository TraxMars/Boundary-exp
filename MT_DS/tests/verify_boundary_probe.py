"""Verify target parity, graph placement, and diagnostic state isolation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from boundary_head import independent_boundary_loss
from networks.vnet_ds import VNetDS
from networks.vnet_boundary_probe import VNetBoundaryProbe
from objectives import supervised_loss
from online_probe import channel_interventions
from probe_objectives import morphology_boundary, probe_boundary_loss


def hashes():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((PROJECT / "code").rglob("*.py"))}


def main():
    torch.set_num_threads(2)
    checks = []
    old_path = PROJECT.parents[1] / "MT+DuCiSC+BGCE/MT-DuCiSC/results/LA_BoundaryProbeW003_8_labeled_0.01_0.99_seed1337/teacher_0.3_none/code/networks/boundary_probe.py"
    spec = importlib.util.spec_from_file_location("historical_probe", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    labels = torch.zeros(2, 32, 32, 32, dtype=torch.long)
    labels[:, 8:24, 8:24, 8:24] = 1
    target = morphology_boundary(labels)
    single = torch.zeros(1, 5, 5, 5, dtype=torch.long)
    single[:, 2, 2, 2] = 1
    assert morphology_boundary(single).sum() == 27
    assert morphology_boundary(torch.ones_like(single)).sum() == 0
    logits = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
    expected, _, _ = old.boundary_loss(logits, target)
    actual, _ = probe_boundary_loss(logits, target, "e4")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    encoded = target[:, 0].long()
    native = torch.randn(2, 1, 16, 16, 16, requires_grad=True)
    expected, _ = independent_boundary_loss((native, native), encoded)
    actual, _ = probe_boundary_loss(native, target, "d3")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    checks.append("historical_E4_loss_and_current_D3_loss_exact_parity_morphology_padding")

    for variant in ("e4", "d3"):
        torch.manual_seed(42)
        plain = VNetDS(1, 2, normalization="batchnorm", has_dropout=True).train()
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        added = VNetBoundaryProbe(1, 2, normalization="batchnorm", has_dropout=True,
                                  variant=variant).train()
        assert torch.equal(baseline_rng, torch.get_rng_state())
        for name, value in plain.state_dict().items():
            torch.testing.assert_close(value, added.state_dict()[name], rtol=0, atol=0)
        image = torch.randn(2, 1, 32, 32, 32)
        rng = torch.get_rng_state().clone()
        plain_outputs = plain(image, deepsup=True)
        rng_after_plain = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        outputs, boundary, features = added(image, deepsup=True, return_boundary=True)
        assert torch.equal(rng_after_plain, torch.get_rng_state())
        for a, b in zip(plain_outputs, outputs):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        before = {k: v.clone() for k, v in added.state_dict().items()}
        rng = torch.get_rng_state().clone()
        modes = [m.training for m in added.modules()]
        diagnostics = channel_interventions(added, features, labels, target,
                                             step=1, seed=42, random_repeats=2,
                                             epsilons=(.01, 1.))
        assert len(diagnostics["interventions"]) == 8
        assert diagnostics["score_norm"] > 0
        assert torch.equal(rng, torch.get_rng_state())
        assert modes == [m.training for m in added.modules()]
        for k, v in added.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        assert all(p.grad is None for p in added.parameters())
        sup_a, _, _ = supervised_loss(plain_outputs, labels)
        sup_b, _, _ = supervised_loss(outputs, labels)
        sup_a.backward()
        sup_b.backward()
        for name, parameter in plain.named_parameters():
            torch.testing.assert_close(parameter.grad, dict(added.named_parameters())[name].grad,
                                       rtol=0, atol=0)
        assert all(p.grad is None for p in added.boundary_head.parameters())
        for model in (plain, added):
            torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4).step()
        for name, value in plain.state_dict().items():
            torch.testing.assert_close(value, added.state_dict()[name], rtol=0, atol=0)
        checks.append(f"{variant}_initialization_logits_RNG_BN_gradients_SGD_preserved_by_readonly_diagnostics")
        added.zero_grad(set_to_none=True)
        _, boundary, _ = added(image, deepsup=True, return_boundary=True)
        value, _ = probe_boundary_loss(boundary, target, variant)
        value.backward()
        assert added.block_one.conv[0].weight.grad.norm() > 0
        assert added.boundary_head[0].weight.grad.norm() > 0
        assert added.block_nine.conv[0].weight.grad is None
        if variant == "e4":
            assert added.block_four.conv[0].weight.grad.norm() > 0
            assert added.block_five.conv[0].weight.grad is None
        else:
            assert added.block_five.conv[0].weight.grad.norm() > 0
            assert added.block_seven_up.conv[0].weight.grad.norm() > 0
        assert not added.out_conv_2._forward_pre_hooks
        assert all(torch.isfinite(p.grad).all() for p in added.parameters() if p.grad is not None)
        checks.append(f"{variant}_boundary_gradient_reaches_only_intended_graph_and_hooks_removed")
        del plain, added, outputs, plain_outputs, features, boundary, before

    report = PROJECT / "verification_runs/boundary_probe_v1/unit_results.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status="PASS", checks=checks, source_hashes=hashes())
    report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", checks=checks, report=str(report)), indent=2))


if __name__ == "__main__":
    main()
