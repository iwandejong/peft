from peft.tuners.lora.model import LoraModel
from .layer import SpikeLoraLayer, replace_linear_with_spikelora

class SpikeLoraModel(LoraModel):
    """
    SpikeLoRA variant of PEFT's LoraModel.
    Replaces target Linear layers with SpikeLoraLayer.
    """

    def __init__(self, model, config, adapter_name="default"):
        super().__init__(model, config, adapter_name)

        # Replace Linear with SpikeLoRA after LoraModel's init
        replace_linear_with_spikelora(
            model=self.base_model,
            target_modules=self.config.target_modules,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            v_threshold=self.config.v_threshold,
        )

        # Collect SpikeLoRA layers
        self.lora_layers = [
            module for module in self.base_model.modules()
            if isinstance(module, SpikeLoraLayer)
        ]
