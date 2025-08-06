import torch
import torch.nn as nn
from typing import List, Optional

from .config import SpikeLoraConfig
from .layer import SpikeLoraLayer, replace_linear_with_lora


class SpikeLoraModel(nn.Module):
    """
    Simple wrapper that adds LoRA to a pretrained model.
    """
    
    def __init__(self, model: nn.Module, config: SpikeLoraConfig):
        super().__init__()
        
        self.base_model = model
        self.config = config
        self.lora_layers = []
        
        # Apply LoRA to target modules
        self._add_lora_layers()
        
        # Only LoRA parameters are trainable
        self._freeze_base_model()
    
    def _add_lora_layers(self):
        """Add LoRA layers to target modules."""
        target_modules = self.config.target_modules
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        
        replace_linear_with_lora(
            model=self.base_model,
            target_modules=target_modules,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        # Collect LoRA layers for easy access
        for module in self.base_model.modules():
            if isinstance(module, SpikeLoraLayer):
                self.lora_layers.append(module)
    
    def _freeze_base_model(self):
        """Freeze base model parameters, keep LoRA parameters trainable."""
        # Freeze all parameters first
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for lora_layer in self.lora_layers:
            for param in lora_layer.lora_A.parameters():
                param.requires_grad = True
            for param in lora_layer.lora_B.parameters():
                param.requires_grad = True
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.base_model(*args, **kwargs)
    
    def merge_and_unload(self):
        """
        Merge LoRA weights and return the base model.
        Useful for inference or saving merged model.
        """
        # Merge all LoRA layers
        for lora_layer in self.lora_layers:
            lora_layer.merge()
        
        # Return the base model (now with LoRA merged in)
        return self.base_model
    
    def get_trainable_params(self):
        """Get number of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def print_trainable_params(self):
        """Print trainable parameter statistics."""
        trainable, total = self.get_trainable_params()
        print(f"Trainable params: {trainable:,} || "
              f"Total params: {total:,} || "
              f"Trainable%: {100 * trainable / total:.4f}%")


def get_peft_model(model: nn.Module, peft_config: SpikeLoraConfig):
    """
    Simple function to wrap a model with LoRA.
    
    Args:
        model: Base model to add LoRA to
        peft_config: LoRA configuration
    
    Returns:
        LoRA-enabled model
    """
    return SpikeLoraModel(model, peft_config)