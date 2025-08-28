# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, surrogate

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class SpikeLoraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("spikelora_A", "spikelora_B", "lora_dropout", "spikelora_lif")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "v_threshold")

    def __init__(self, base_layer: nn.Module, **kwargs):
        super().__init__()  # Call BaseTunerLayer.__init__
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.spikelora_A = nn.ParameterDict({})
        self.spikelora_B = nn.ParameterDict({})
        self.lora_dropout = nn.ModuleDict({})
        self.spikelora_lif = nn.ModuleDict({})
        self.v_threshold = {}

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, 'weight') and hasattr(base_layer.weight, 'shape'):
            if len(base_layer.weight.shape) == 2:
                in_features, out_features = base_layer.weight.shape
            else:
                raise ValueError(f"Unsupported base layer type: {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        v_threshold: float = 0.5,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        self.scaling[adapter_name] = lora_alpha / r
        self.v_threshold[adapter_name] = v_threshold

        self.spikelora_A[adapter_name] = nn.Parameter(torch.randn(self.in_features, r))
        self.spikelora_B[adapter_name] = nn.Parameter(torch.randn(r, self.out_features))
        self.lora_dropout[adapter_name] = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.spikelora_lif[adapter_name] = neuron.ParametricLIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            v_threshold=v_threshold,
            detach_reset=True,
        )

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.spikelora_A.keys():
            nn.init.normal_(self.spikelora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.spikelora_B[adapter_name], mean=0.0, std=0.02)


class Linear(nn.Module, SpikeLoraLayer):
    # SpikeLoRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        v_threshold: float = 0.5,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        # this gets the init from nn.Module's super perspective, which should always be called
        super().__init__()
        SpikeLoraLayer.__init__(self, base_layer, **kwargs)

        # Freezing the pre-trained weight matrix
        self.get_base_layer().weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, v_threshold)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            base_layer = self.get_base_layer()
            if active_adapter in self.spikelora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.spikelora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        if adapter not in self.spikelora_A.keys() or self.r[adapter] == 0:
            return torch.zeros_like(self.get_base_layer().weight)
        
        device = self.spikelora_A[adapter].device
        dtype = self.spikelora_A[adapter].dtype
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16
        
        if cast_to_fp32:
            lora_A = self.spikelora_A[adapter].float()
            lora_B = self.spikelora_B[adapter].float()
        else:
            lora_A = self.spikelora_A[adapter]
            lora_B = self.spikelora_B[adapter]
        
        # For spiking LoRA, we need to consider the spiking behavior
        # Here we use a simplified approach - in practice you might want to calibrate this
        scaling = self.scaling[adapter]
        delta_weight = scaling * (lora_B @ lora_A)
        
        if self.fan_in_fan_out:
            delta_weight = delta_weight.transpose(0, 1)
        
        return delta_weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.spikelora_A.keys() or self.r[active_adapter] == 0:
                    continue
                
                # Get adapter components
                lora_A = self.spikelora_A[active_adapter]
                lora_B = self.spikelora_B[active_adapter]
                lora_dropout = self.lora_dropout[active_adapter]
                lif = self.spikelora_lif[active_adapter]
                scaling = self.scaling[active_adapter]
                
                # Apply LoRA with spiking
                lora_out = x @ lora_A
                lora_out = lora_dropout(lora_out)
                
                # Reset LIF for each forward pass (pointwise spiking)
                lif.reset()
                spikes = lif(lora_out)
                lora_out = lora_out * spikes  # gated by spikes
                
                lora_out = lora_out @ lora_B * scaling
                result = result + lora_out
        
        result = result.to(previous_dtype)
        return result
    
    def __repr__(self) -> str:
        rep = super().__repr__()
        return "spikelora." + rep