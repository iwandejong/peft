#!/bin/bash
for TASK in stsb sst2 mnli qqp
do
for LORA in "" "--lora"
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK $LORA > logs/${TASK}_${LORA//--/--}.log 2>&1
done
done