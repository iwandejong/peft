from peft.tuners.lora import LoraConfig
from peft.utils import PeftType

class SpikeLoraConfig(LoraConfig):
    """Configuration class for SpikeLoRA, inheriting from LoraConfig.
    This class extends the LoRA configuration to include a spiking activation threshold.
    Args:
        v_threshold (float): The spiking activation threshold.
        **kwargs: Additional keyword arguments for LoRA configuration.
    """
    def __init__(self, v_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.v_threshold = v_threshold
        self.peft_type = PeftType.SPIKELORA
