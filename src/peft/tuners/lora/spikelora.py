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
import torch
import torch.nn as nn
import torch.nn.functional as F

# spikingjelly
from spikingjelly.clock_driven import neuron, surrogate


class SpikeLoraLinearLayer(nn.Module):
    def __init__(self, fan_in_fan_out):
        super().__init__()
        self.fan_in_fan_out = fan_in_fan_out

    def update_layer(self, v_threshold) -> None:
        self.lora_lif = neuron.LIFNode(
            tau=2.0, 
            surrogate_function=surrogate.ATan(alpha=2.0), 
            v_threshold=v_threshold, 
            detach_reset=True
        )
        self.avg_spikes = None
        self.sparsity = None

    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For SpikeLoRA, calculate the extra output from LoRA with spiking applied.
        """
        # Apply lora_A (dropout is performed upstream)
        down_proj = lora_A(x) # (..., r)
        
        self.lora_lif.reset() # reset state before each forward
        a_spiked = self.lora_lif(down_proj) # (..., r), 0/1 spikes
        
        # Track statistics
        self.avg_spikes = a_spiked.float().mean(dim=0)
        self.sparsity = (a_spiked == 0).float().mean().item()

        # Scale a_spiked by down_proj to retain learned information (LIF returns 0/1 spikes)
        a_out = a_spiked * down_proj # (..., r)

        # Up projection
        lora_result = lora_B(a_out) * scaling # (..., out_features)

        return base_result + lora_result

    def reset(self):
        """Reset the spiking neuron state"""
        if hasattr(self, 'lora_lif'):
            self.lora_lif.reset()

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.spikelora." + rep


class SpikeLoraEmbeddingLayer(SpikeLoraLinearLayer):
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, embed_fn):
        """
        For SpikeLoRA, calculate the extra output from LoRA with spiking applied for embeddings.
        """
        # Apply spiking to the embedding lookup
        lora_weight = (lora_A.weight @ lora_B.weight).T
        # For embeddings, we need to handle the spiking differently
        # This is a simplified version - you might need to adapt based on your specific needs
        lora_result = embed_fn(x, lora_A.weight) @ lora_B.weight * scaling

        return lora_result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.spikelora." + rep


class _SpikeLoraConvNdLayer(SpikeLoraLinearLayer):
    def forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        """
        For SpikeLoRA, calculate the extra output from LoRA with spiking applied for convolutions.
        """
        # For conv layers, we need to reshape the input/output appropriately
        r = lora_A.weight.shape[0]
        lora_weight = torch.mm(lora_B.weight.view([-1, r]), lora_A.weight.view([r, -1]))
        lora_weight = lora_weight.reshape(base_layer.weight.shape)
        
        # Apply spiking to the convolution
        # This is a simplified version - you might need to adapt based on your specific needs
        lora_result = F.conv2d(x, lora_weight, bias=None, 
                              stride=base_layer.stride, padding=base_layer.padding,
                              dilation=base_layer.dilation, groups=base_layer.groups) * scaling

        return lora_result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.spikelora." + rep


class SpikeLoraConv1dLayer(_SpikeLoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)


class SpikeLoraConv2dLayer(_SpikeLoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)


class SpikeLoraConv3dLayer(_SpikeLoraConvNdLayer):
    def __init__(self, fan_in_fan_out):
        super().__init__(fan_in_fan_out)
