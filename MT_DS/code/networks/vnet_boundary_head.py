"""Original VNetDS with two training-only boundary classification heads."""
import torch
from torch import nn

from .vnet_ds import VNetDS


class VNetBoundaryHead(VNetDS):
    def __init__(self, *args, boundary_head_seed=1000045, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the backbone's RNG stream, including the first training batch.
        # Seed only the CPU generator; torch.manual_seed also changes CUDA RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(boundary_head_seed)
            self.boundary_half = self.make_head(self.out_conv_2.in_channels)
            self.boundary_full = self.make_head(self.out_conv.in_channels)

    @staticmethod
    def make_head(channels):
        hidden = min(channels, 32)
        return nn.Sequential(nn.Conv3d(channels, hidden, 3, padding=1),
                             nn.ReLU(inplace=False), nn.Conv3d(hidden, 1, 1))

    def forward(self, x, deepsup=False, return_boundary=False):
        if not return_boundary:
            return super().forward(x, deepsup=deepsup)
        # Observe exactly the inputs used by the existing segmentation heads.
        # Local hooks avoid copying or modifying the frozen decoder definition.
        features = {}

        def capture(name):
            def hook(module, inputs):
                features[name] = inputs[0]
            return hook

        hooks = [self.out_conv_2.register_forward_pre_hook(capture("half")),
                 self.out_conv.register_forward_pre_hook(capture("full"))]
        try:
            segmentation = super().forward(x, deepsup=deepsup)
        finally:
            for hook in hooks:
                hook.remove()
        boundaries = (self.boundary_half(features["half"]),
                      self.boundary_full(features["full"]))
        return segmentation, boundaries
