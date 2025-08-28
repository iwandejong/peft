#!/usr/bin/env python3
"""
Simple test script to verify SpikeLoRA implementation
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

def test_spikelora():
    """Test basic SpikeLoRA functionality"""
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    print("Original model:")
    print(model)
    
    # Create SpikeLoRA config (enabled by default)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["0", "2"],  # Target the Linear layers
        use_spikelora=True,  # This should be True by default now
        spikelora_v_threshold=0.01
    )
    
    print(f"\nSpikeLoRA config:")
    print(f"use_spikelora: {config.use_spikelora}")
    print(f"spikelora_v_threshold: {config.spikelora_v_threshold}")
    
    # Apply SpikeLoRA
    peft_model = get_peft_model(model, config)
    
    print(f"\nPEFT model:")
    print(peft_model)
    
    # Test forward pass
    x = torch.randn(2, 10)
    output = peft_model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Check if SpikeLoRA layers are present
    has_spike_layers = False
    for name, module in peft_model.named_modules():
        if "lora_spike_layer" in name:
            has_spike_layers = True
            print(f"Found SpikeLoRA layer: {name}")
            print(f"Module type: {type(module)}")
    
    if has_spike_layers:
        print("\n✅ SpikeLoRA layers successfully added!")
    else:
        print("\n❌ No SpikeLoRA layers found!")
    
    return has_spike_layers

if __name__ == "__main__":
    try:
        success = test_spikelora()
        if success:
            print("\n🎉 SpikeLoRA test passed!")
        else:
            print("\n💥 SpikeLoRA test failed!")
    except Exception as e:
        print(f"\n💥 Error during test: {e}")
        import traceback
        traceback.print_exc()
