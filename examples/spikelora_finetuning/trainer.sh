#!/bin/bash
for LORA in "" "--lora"
do
echo "Running: python3 examples/spikelora_finetuning/spikelora_glue.py --quantize $LORA"
python3 examples/spikelora_finetuning/spikelora_glue.py --quantize $LORA
done