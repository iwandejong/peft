import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate
from peft.tuners.tuners_utils import BaseTunerLayer

import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate
from peft.tuners.tuners_utils import BaseTunerLayer

class SpikeLoraLayer(BaseTunerLayer):
    """
    LoRA on top of a base nn.Linear, with a spiking gate in the LoRA branch.
    Note: Nonlinear path => true merge is not mathematically valid; we disable it.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        v_threshold: float = 0.5,
        adapter_name: str = "default",
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("SpikeLoraLayer expects nn.Linear as base_layer")

        # keep a real Linear with the same params (including bias)
        self.base = nn.Linear(
            base_layer.in_features,
            base_layer.out_features,
            bias=(base_layer.bias is not None),
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        with torch.no_grad():
            self.base.weight.copy_(base_layer.weight.data)
            if self.base.bias is not None and base_layer.bias is not None:
                self.base.bias.copy_(base_layer.bias.data)

        self.in_features = self.base.in_features
        self.out_features = self.base.out_features

        # multi-adapter plumbing expected by BaseTunerLayer / LoraModel
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.lif = nn.ModuleDict()

        self.weight = self.base.weight  # for compatibility with some downstream checks
        self.bias = self.base.bias

        self.set_adapter(
            adapter_name,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            v_threshold=v_threshold,
            init_lora_weights=init_lora_weights,
        )
        self.active_adapter = adapter_name

        # merging is not supported due to nonlinearity
        self._merged = False
        self._merge_supported = False

        # by default, freeze base params; training only LoRA
        for p in self.base.parameters():
            p.requires_grad = False

    def set_adapter(
        self,
        adapter_name: str,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        v_threshold: float = 0.5,
        init_lora_weights: bool = True,
    ):
        if r <= 0:
            # still register no-op to keep PEFT happy
            self.r[adapter_name] = 0
            self.lora_alpha[adapter_name] = 0
            self.scaling[adapter_name] = 0.0
            self.lora_A[adapter_name] = nn.Identity()
            self.lora_B[adapter_name] = nn.Identity()
            self.lora_dropout[adapter_name] = nn.Identity()
            self.lif[adapter_name] = nn.Identity()
            return

        dev = self.base.weight.device
        dt = self.base.weight.dtype

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r

        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False, device=dev, dtype=dt)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False, device=dev, dtype=dt)
        self.lora_dropout[adapter_name] = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lif[adapter_name] = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            v_threshold=v_threshold,
            detach_reset=True,
        )

        if init_lora_weights:
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)

        # train only LoRA params for this adapter
        for p in self.lora_A[adapter_name].parameters():
            p.requires_grad = True
        for p in self.lora_B[adapter_name].parameters():
            p.requires_grad = True

    def _get_active(self):
        a = self.active_adapter
        return (
            self.lora_A[a],
            self.lora_B[a],
            self.lora_dropout[a],
            self.lif[a],
            self.scaling[a],
            self.r[a],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        a, b, do, lif, scaling, r = self._get_active()
        if r == 0 or self._merged:
            return base_out

        # pointwise spiking in LoRA branch (no temporal carry-over)
        # if you want temporal dynamics, do NOT reset; otherwise this keeps it a static nonlinearity
        lif.reset()

        lora_out = a(x)
        lora_out = do(lora_out)
        spikes = lif(lora_out)
        lora_out = lora_out * spikes  # gated
        lora_out = b(lora_out) * scaling
        return base_out + lora_out

    # ---- merge API (disabled) ----
    def merge(self):
        if not self._merge_supported:
            raise RuntimeError(
                "SpikeLoraLayer.merge() is not supported: LoRA branch is nonlinear (spiking). "
                "Move the LIF outside the LoRA path or set approximate_merge=True with calibration."
            )
        self._merged = True

    def unmerge(self):
        self._merged = False

    # optional: approximate merge via firing-rate calibration (use at your own risk)
    @torch.no_grad()
    def approximate_merge(self, dataloader, steps: int = 256):
        """
        Estimate an average gate (0..1) per LoRA neuron and fold it into B.
        WARNING: Input-dependent; only a heuristic.
        """
        a, b, do, lif, scaling, r = self._get_active()
        if r == 0:
            return
        dev = self.base.weight.device
        total = torch.zeros(r, device=dev)
        count = 0
        lif.train(False)
        for i, (xb, *_) in enumerate(dataloader):
            if i >= steps: break
            xb = xb.to(dev)
            z = a(xb)
            z = do(z)
            s = lif(z)  # in {0,1} via surrogate
            total += s.mean(dim=0)
            count += 1
        gate = (total / max(count, 1)).clamp_(0, 1)  # r-dim
        # fold: B <- B * diag(gate), W_eff <- W + scaling * B A
        Bg = b.weight * gate.view(-1, 1)
        self.base.weight.data += scaling * (Bg @ a.weight)
        self._merged = True


import re
from typing import Iterable, Union, Pattern

def _match(name: str, targets: Iterable[Union[str, Pattern]]) -> bool:
    for t in targets:
        if isinstance(t, str) and name.endswith(t):
            return True
        if hasattr(t, "search") and t.search(name):
            return True
    return False

def replace_linear_with_spikelora(
    model: nn.Module,
    target_modules: Iterable[Union[str, Pattern]],
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    v_threshold: float = 0.5,
    adapter_name: str = "default",
):
    """
    Replace selected nn.Linear layers with SpikeLoraLayer.
    Matches by exact suffix or regex Pattern.
    """
    to_replace = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match(module_name, target_modules):
            # find parent
            parent = model
            parts = module_name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
            to_replace.append((parent, leaf, module))

    for parent, leaf, linear in to_replace:
        sp = SpikeLoraLayer(
            base_layer=linear,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            v_threshold=v_threshold,
            adapter_name=adapter_name,
        ).to(linear.weight.device).to(linear.weight.dtype)
        setattr(parent, leaf, sp)
