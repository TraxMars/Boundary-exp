"""MT_DS with either the historical E4 probe or a single D3 readout."""
import torch
from torch import nn
import torch.nn.functional as F

from .vnet_ds import VNetDS


class VNetBoundaryProbe(VNetDS):
    def __init__(self, *args, variant, boundary_head_seed=1000045, **kwargs):
        super().__init__(*args, **kwargs)
        if variant not in ("e4", "d3"):
            raise ValueError(variant)
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(boundary_head_seed)
            if variant == "e4":
                self.boundary_head = nn.Sequential(
                    nn.Conv3d(128, 32, 3, padding=1, bias=False),
                    nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.Conv3d(32, 1, 1))
            else:
                self.boundary_head = nn.Sequential(
                    nn.Conv3d(32, 32, 3, padding=1),
                    nn.ReLU(inplace=False), nn.Conv3d(32, 1, 1))

    def decode_probe(self, features, gate=None):
        """Gate only a skip copy; deeper cached encoder features stay unchanged."""
        branch = list(features)
        index = 3 if self.variant == "e4" else 1
        if gate is not None:
            branch[index] = branch[index] * gate[..., None, None, None]
        observed = {}

        def capture(module, inputs):
            observed["half"] = inputs[0]

        hook = self.out_conv_2.register_forward_pre_hook(capture) if self.variant == "d3" else None
        try:
            outputs = self.decoder(branch)
        finally:
            if hook is not None:
                hook.remove()
        feature = branch[3] if self.variant == "e4" else observed["half"]
        logits = self.boundary_head(feature)
        if self.variant == "e4":
            logits = F.interpolate(logits, size=features[0].shape[2:],
                                   mode="trilinear", align_corners=False)
        return outputs, logits, feature

    def forward(self, x, deepsup=False, return_boundary=False):
        if not return_boundary:
            return super().forward(x, deepsup=deepsup)
        features = self.encoder(x)
        outputs, logits, _ = self.decode_probe(features)
        return outputs, logits, tuple(f.detach() for f in features)
