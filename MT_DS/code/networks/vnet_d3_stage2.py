"""D3 readout-only control and state-isolated Student L-to-U tail replay."""
import torch
from torch import nn
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .vnet_boundary_probe import VNetBoundaryProbe


class DecoderTail(nn.Module):
    """An ephemeral view; never register this alias inside the owning Student."""
    def __init__(self, model):
        super().__init__()
        for name in ("out_conv_2", "block_eight", "block_eight_up", "block_nine", "dropout", "out_conv"):
            setattr(self, name, getattr(model, name))
        self.has_dropout = model.has_dropout

    def forward(self, half_feature, first_skip):
        half = self.out_conv_2(half_feature)
        feature = self.block_eight_up(self.block_eight(half_feature)) + first_skip
        feature = self.block_nine(feature)
        if self.has_dropout:
            feature = self.dropout(feature)
        return half, self.out_conv(feature)


class VNetD3Stage2(VNetBoundaryProbe):
    def __init__(self, *args, readout_only=False, interaction=False,
                 transfer_seed=2000045, **kwargs):
        super().__init__(*args, variant="d3", **kwargs)
        self.readout_only = readout_only
        if interaction:
            from d3_transfer import BoundaryPrototypeTransfer
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(transfer_seed)
                self.transfer = BoundaryPrototypeTransfer(32)

    def _forward_observed(self, x, need_context):
        features = self.encoder(x)
        observed = {}

        def capture_half(module, inputs):
            observed["half"] = inputs[0]
            if need_context:
                observed["buffers"] = {n: b.detach().clone() for n,b in DecoderTail(self).named_buffers()}

        def capture_dropout(module, inputs):
            # This hook is installed after encoder(), so it sees only decoder dropout.
            observed["cpu_rng"] = torch.get_rng_state().clone()
            observed["cuda_rng"] = (torch.cuda.get_rng_state(x.device).clone() if x.is_cuda else None)

        hooks = [self.out_conv_2.register_forward_pre_hook(capture_half)]
        if need_context and self.has_dropout:
            hooks.append(self.dropout.register_forward_pre_hook(capture_dropout))
        try:
            outputs = self.decoder(features)
        finally:
            for hook in hooks:
                hook.remove()
        return outputs, features, observed

    def forward(self, x, deepsup=False, return_boundary=False):
        if not return_boundary:
            return super().forward(x, deepsup=deepsup)
        outputs, features, observed = self._forward_observed(x, False)
        head_input = observed["half"].detach() if self.readout_only else observed["half"]
        logits = self.boundary_head(head_input)
        return outputs, logits, tuple(f.detach() for f in features)

    def forward_context(self, x):
        outputs, features, context = self._forward_observed(x, True)
        context["features"] = features
        return outputs, context

    def replay_tail(self, changed_half, context, checkpoint_tail=True):
        first_skip = context["features"][0]

        def replay(half, skip):
            adapter = DecoderTail(self)
            params = dict(adapter.named_parameters())
            # Functional BN mutates these copies, never the live Student buffers.
            buffers = {n:b.clone() for n,b in context["buffers"].items()}
            devices = [half.device.index or 0] if half.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                if self.has_dropout:
                    torch.set_rng_state(context["cpu_rng"])
                    if half.is_cuda:
                        torch.cuda.set_rng_state(context["cuda_rng"], half.device)
                return functional_call(adapter, (params, buffers), (half, skip), strict=True)

        if checkpoint_tail and torch.is_grad_enabled():
            return checkpoint(replay, changed_half, first_skip, use_reentrant=False,
                              preserve_rng_state=True)
        return replay(changed_half, first_skip)
