import math
import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate

class SpikeLoraLayer(nn.Module):
    """SpikeLoRA layer that applies LoRA to a base linear layer with spiking activation.
    Args:
        base_layer: The original linear layer to apply LoRA to.
        r: LoRA rank (default: 8)
        lora_alpha: LoRA scaling parameter (default: 8)
        lora_dropout: Dropout probability (default: 0.0)
        init_lora_weights: Whether to initialize LoRA weights (default: True)
        v_threshold: Threshold for spike activation (default: 0.5)
    Attributes:
        base_layer: The original linear layer.
        r: LoRA rank.
        lora_alpha: LoRA scaling parameter.
        scaling: Scaling factor for LoRA weights.
        lora_A: LoRA weight matrix A.
        lora_B: LoRA weight matrix B.
        lora_A_lif: Spiking neuron for LoRA A.
        lora_dropout: Dropout layer for LoRA A.
        merged: Whether the LoRA weights have been merged into the base layer.
    Methods:
        forward: Forward pass through the SpikeLoRA layer.
        merge: Merge LoRA weights into the base layer.
        unmerge: Unmerge LoRA weights from the base layer.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        v_threshold: float = 0.5
    ):
        super().__init__()

        if base_layer.bias is not None:
            raise ValueError("SpikeLoRA does not support bias in the base layer. Please set bias=None in the base layer.")

        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # LoRA matrices
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)

        # SpikeLoRA neurons
        self.lora_A_lif = neuron.LIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0), v_threshold=v_threshold, detach_reset=True)

        # Dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # Initialize weights
        if init_lora_weights:
            self._init_weights()

        # Merge state
        self.merged = False

    def _init_weights(self):
        """Initialize LoRA weights following standard practice."""
        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SpikeLoRA layer.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after applying LoRA and spiking activation.
        """
        # Reset spiking potential
        self.lora_A_lif.reset()

        # Base layer output
        base_output = self.base_layer(x)

        if self.merged:
            # If merged, LoRA is already in base layer weights
            return base_output

        # LoRA path: x -> A -> dropout -> SpikeLoRA -> B -> scale
        lora_output = self.lora_A(x)
        lora_output = self.lora_dropout(lora_output)

        # SpikeLORA
        lora_spikes = self.lora_A_lif(lora_output) # SpikeLoRA activation
        lora_output = lora_output * lora_spikes

        lora_output = self.lora_B(lora_output)
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge(self):
        """Merge LoRA weights into the base layer."""
        if self.merged:
            return

        lora_weight = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.base_layer.weight.data += lora_weight
        self.merged = True

    def unmerge(self):
        """Unmerge LoRA weights from the base layer."""
        if not self.merged:
            return

        lora_weight = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.base_layer.weight.data -= lora_weight
        self.merged = False


def replace_linear_with_lora(
    model: nn.Module,
    target_modules: list,
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0,
    v_threshold: float = 0.5
):
    """
    Helper to replace Linear layers with LoRA layers.

    Args:
        model: The model to modify
        target_modules: List of module names to replace (e.g., ['q_proj', 'v_proj'])
        r: LoRA rank
        lora_alpha: LoRA scaling parameter
        lora_dropout: LoRA dropout probability
        v_threshold: Threshold for spike activation
    """

    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent = model
                names = name.split('.')
                for n in names[:-1]:
                    parent = getattr(parent, n)

                # Replace with LoRA layer
                lora_layer = SpikeLoraLayer(
                    base_layer=module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    v_threshold=v_threshold
                )

                setattr(parent, names[-1], lora_layer)