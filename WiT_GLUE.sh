#!/bin/bash

# GLUE
for TASK in mrpc stsb rte sst2 qnli qqp mnli cola
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --project=WiT"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --project=WiT > logs/${TASK}.log 2>&1
done
done

# ranks
for TASK in cola
do
for RANK in 1 2 4 8 16
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --rank $RANK --project=WiT"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --rank $RANK $LORA --project=WiT > logs/${TASK}_r${RANK}.log 2>&1
done
done

# lr
for TASK in cola
do
for LR in 0.0001 0.0005 0.0007 0.0009
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --lr $LR --project=WiT"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --lr $LR --project=WiT > logs/${TASK}_lr${LR//./-}.log 2>&1
done
done