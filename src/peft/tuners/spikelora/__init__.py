"""
SpikeLoRA Implementation for PEFT
"""

from peft.utils import register_peft_method

from .config import SpikeLoraConfig
from .layer import Linear, SpikeLoraLayer
from .model import SpikeLoraModel

__all__ = [
    "Linear",
    "SpikeLoraConfig",
    "SpikeLoraLayer", 
    "SpikeLoraModel"
]

register_peft_method(
    name="spikelora",
    config_cls=SpikeLoraConfig,
    model_cls=SpikeLoraModel
)