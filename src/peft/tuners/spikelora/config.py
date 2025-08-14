from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class SpikeLoraConfig(PeftConfig):
    """
    Configuration for SpikeLoRA.
    
    Args:
        r: LoRA rank (default: 8)
        target_modules: Names of modules to apply LoRA to
        lora_alpha: LoRA scaling parameter (default: 8)  
        lora_dropout: Dropout probability (default: 0.0)
        init_lora_weights: Whether to initialize LoRA weights (default: True)
        v_threshold: Threshold for spike activation (default: 0.5)
    """
    
    r: int = field(default=8, metadata={"help": "LoRA rank"})
    target_modules: Optional[Union[List[str], str]] = field(default=None, metadata={"help": "Target module names for LoRA"})
    lora_alpha: int = field(default=8, metadata={"help": "LoRA scaling parameter"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    init_lora_weights: bool = field(default=True, metadata={"help": "Initialize LoRA weights"})
    v_threshold: float = field(default=0.5, metadata={"help": "Threshold for spike activation"})

    def __post_init__(self):
        self.peft_type = PeftType.BASIC_LORA