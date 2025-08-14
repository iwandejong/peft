import torch.nn as nn

from .config import SpikeLoraConfig
from .layer import SpikeLoraLayer, replace_linear_with_lora

class SpikeLoraModel(nn.Module):
    """
    SpikeLoRA model that applies LoRA to specified modules in a base model.
    Args:
        model: Pretrained model to apply LoRA to.
        config: Configuration for SpikeLoRA.
    Attributes:
        base_model: The original model with LoRA layers added.
        config: Configuration for SpikeLoRA.
        lora_layers: List of LoRA layers added to the model.
    Methods:
        forward: Forward pass through the model.
        merge_and_unload: Merge LoRA weights into the base model for inference.
        get_trainable_params: Get number of trainable parameters.
        print_trainable_params: Print number of trainable parameters.
    """
    
    def __init__(self, model: nn.Module, config: SpikeLoraConfig):
        """
        Args:
            model: Pretrained model to apply LoRA to.
            config: Configuration for SpikeLoRA.
        """
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
        """
        Forward pass through the model.
        Args:
            *args: Positional arguments for the base model.
            **kwargs: Keyword arguments for the base model.
        Returns:
            Output of the base model with SpikeLoRA applied.
        """
        return self.base_model(*args, **kwargs)
    
    def merge_and_unload(self):
        """
        Merge LoRA weights and return the base model.
        This is useful for inference after training.
        Returns:
            nn.Module: The base model with LoRA weights merged.
        """
        # Merge all LoRA layers
        for lora_layer in self.lora_layers:
            lora_layer.merge()
        
        # Return the base model (now with LoRA merged in)
        return self.base_model
    
    def get_trainable_params(self):
        """
        Get number of trainable parameters.
        Returns:
            Tuple of (trainable parameters, total parameters).
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def print_trainable_params(self):
        """Print number of trainable parameters."""
        trainable, total = self.get_trainable_params()
        print(f"Trainable params: {trainable:,} || "
              f"Total params: {total:,} || "
              f"Trainable%: {100 * trainable / total:.4f}%")


def get_peft_model(model: nn.Module, peft_config: SpikeLoraConfig):
    """
    Get a PEFT model with SpikeLoRA configuration.
    Args:
        model: Pretrained model to apply LoRA to.
        peft_config: Configuration for SpikeLoRA.
    Returns:
        SpikeLoraModel: Model with LoRA layers added.
    """
    return SpikeLoraModel(model, peft_config)