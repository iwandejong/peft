from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.layer import ParamWrapper
from peft.utils.other import get_pattern_key
from .layer import Linear, SpikeLoraLayer
import torch
from typing import Optional
import warnings

class SpikeLoraModel(LoraModel):
    """
    SpikeLoRA variant of PEFT's LoraModel.
    Replaces target Linear layers with SpikeLoraLayer.
    """
    prefix: str = "spikelora_"

    def _create_and_replace(
        self,
        spikelora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {}
        kwargs["bias"] = bias

        # Extract parameters from spikelora_config
        kwargs.update({
            "r": spikelora_config.r,
            "lora_alpha": spikelora_config.lora_alpha,
            "lora_dropout": spikelora_config.lora_dropout,
            "init_lora_weights": spikelora_config.init_lora_weights,
            "v_threshold": spikelora_config.v_threshold,
        })

        # Add any additional optional kwargs
        for k, v in optional_kwargs.items():
            kwargs[k] = v

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                spikelora_config.r,
                lora_alpha=spikelora_config.lora_alpha,
                lora_dropout=spikelora_config.lora_dropout,
                init_lora_weights=spikelora_config.init_lora_weights,
                v_threshold=spikelora_config.v_threshold,
            )
        else:
            new_module = self._create_new_module(spikelora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # For SpikeLoRA, we only support Linear layers currently
        if isinstance(target, torch.nn.Linear):
            return Linear(
                base_layer=target,
                adapter_name=adapter_name,
                r=kwargs["r"],
                lora_alpha=kwargs["lora_alpha"],
                lora_dropout=kwargs["lora_dropout"],
                init_lora_weights=kwargs["init_lora_weights"],
                v_threshold=kwargs["v_threshold"],
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported by SpikeLoRA. Currently, only `torch.nn.Linear` is supported."
            )

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, SpikeLoraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name
