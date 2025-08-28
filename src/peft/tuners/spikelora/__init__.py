"""
SpikeLoRA Implementation for PEF
"""

from peft.utils import register_peft_method

from .config import SpikeLoraConfig
from .layer import SpikeLoraLayer, replace_linear_with_lora
from .model import SpikeLoraModel, get_peft_model

__all__ = [
    "SpikeLoraConfig",
    "SpikeLoraLayer", 
    "SpikeLoraModel",
    "get_peft_model",
    "replace_linear_with_lora"
]

register_peft_method(name="spikelora", config_cls=SpikeLoraConfig, model_cls=SpikeLoraModel)