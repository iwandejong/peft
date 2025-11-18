#!/bin/bash
# cola done already
for TASK in mrpc stsb rte sst2 qnli qqp mnli
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK > logs/${TASK}.log 2>&1
done
done