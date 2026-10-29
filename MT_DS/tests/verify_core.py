"""Check backbone equivalence, deep supervision gradients and Mean Teacher."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys

import torch
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from networks import VNetDS
from objectives import consistency_loss, supervised_loss, update_ema
from evaluate import predict_volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=(
        PROJECT.parent / "DuCiSC-main/DuCiSC-main/code/networks/VNet.py"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location("original_vnet", args.original)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    original = original_module.DuCiSC(n_channels=1, n_classes=2, normalization="batchnorm").eval()
    model = VNetDS(n_channels=1, n_classes=2, normalization="batchnorm").eval()
    model.load_state_dict(original.state_dict(), strict=True)
    inputs = torch.randn(2, 1, 32, 32, 32)
    with torch.no_grad():
        expected, _ = original(inputs, deepsup=True)
        actual = model(inputs, deepsup=True)
        assert [tuple(x.shape[2:]) for x in actual] == [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), expected[-1], rtol=0, atol=0)
    print("PASS: exact original predictions and parameter keys at all four scales")
    del original, expected, actual

    teacher = copy.deepcopy(model).requires_grad_(False)
    # A teacher target passed with requires_grad must still be isolated by the loss.
    student_logits = [torch.randn(2, 2, 4, 4, 4, requires_grad=True)]
    teacher_logits = [torch.randn(2, 2, 4, 4, 4, requires_grad=True)]
    consistency_loss(student_logits, teacher_logits).backward()
    assert student_logits[0].grad is not None and teacher_logits[0].grad is None
    print("PASS: consistency sends gradients only to the student")

    model.train()
    teacher.train()
    labels = torch.randint(0, 2, (2, 32, 32, 32))
    outputs = model(inputs, deepsup=True)
    with torch.no_grad():
        targets = teacher(inputs + 0.1, deepsup=True)
    supervised, _, _ = supervised_loss(outputs, labels)
    consistency = consistency_loss(outputs, targets)
    objective = supervised + 0.3 * consistency
    objective.backward()
    for head in (model.out_conv, model.out_conv_2, model.out_conv_3, model.out_conv_4):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
        assert head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    update_ema(model, teacher, 0.99, 0)
    for left, right in zip(model.parameters(), teacher.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    print("PASS: supervised and consistency gradients, all four heads, and first EMA update")

    class SignModel(torch.nn.Module):
        def forward(self, image):
            return torch.cat([-image, image], dim=1)

    for shape in ((9, 12, 7), (22, 25, 19)):
        image = np.random.default_rng(42).normal(size=shape).astype(np.float32)
        prediction = predict_volume(SignModel(), image, (16, 16, 16), (8, 8, 8), "cpu")
        np.testing.assert_array_equal(prediction, image > 0)
    print("PASS: sliding-window overlap, padding and restored volume dimensions")
    print("All core verification checks passed.")


if __name__ == "__main__":
    main()
