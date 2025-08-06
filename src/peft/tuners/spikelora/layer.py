import math
import torch
import torch.nn as nn


class SpikeLoraLayer(nn.Module):
    """
    Simple SpikeLoRA layer that can be applied to Linear layers.
    """
    
    def __init__(
        self, 
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # LoRA matrices
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        
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
        """Forward pass through LoRA layer."""
        # Base layer output
        base_output = self.base_layer(x)
        
        if self.merged:
            # If merged, LoRA is already in base layer weights
            return base_output
        
        # LoRA path: x -> A -> dropout -> B -> scale
        lora_output = self.lora_A(x)
        lora_output = self.lora_dropout(lora_output)
        lora_output = self.lora_B(lora_output)
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output
    
    def merge(self):
        """Merge LoRA weights into base layer for inference."""
        if self.merged:
            return
            
        # Compute LoRA weight: W_lora = scaling * B @ A
        lora_weight = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        
        # Add to base layer weight
        self.base_layer.weight.data += lora_weight
        self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from base layer."""
        if not self.merged:
            return
            
        # Compute LoRA weight and subtract from base layer
        lora_weight = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        self.base_layer.weight.data -= lora_weight
        self.merged = False


def replace_linear_with_lora(
    model: nn.Module, 
    target_modules: list,
    r: int = 8,
    lora_alpha: int = 8,
    lora_dropout: float = 0.0
):
    """
    Simple helper to replace Linear layers with LoRA layers.
    
    Args:
        model: The model to modify
        target_modules: List of module names to replace (e.g., ['q_proj', 'v_proj'])
        r: LoRA rank
        lora_alpha: LoRA scaling parameter  
        lora_dropout: LoRA dropout probability
    """
    
    for name, module in model.named_modules():
        # Check if this module should be replaced
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module
                parent = model
                names = name.split('.')
                for n in names[:-1]:
                    parent = getattr(parent, n)
                
                # Replace with LoRA layer
                lora_layer = SpikeLoraLayer(
                    base_layer=module,
                    r=r,
                    lora_alpha=lora_alpha, 
                    lora_dropout=lora_dropout
                )
                
                setattr(parent, names[-1], lora_layer)