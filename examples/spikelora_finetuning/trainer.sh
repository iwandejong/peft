#!/bin/bash
for QUANTIZE in "" "--quantize"
do
for LORA in "" "--use_spikelora"
do
echo "Running: python3 examples/spikelora_finetuning/spikelora_alpaca.py $QUANTIZE $LORA"
python3 examples/spikelora_finetuning/spikelora_alpaca.py $QUANTIZE $LORA
done
done