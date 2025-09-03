#!/bin/bash
for TASK in cola mrpc stsb rte sst2 qnli qqp mnli
do
for LORA in "" "--lora"
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK $LORA > logs/${TASK}_${LORA//--/--}.log 2>&1
done
done