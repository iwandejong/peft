#!/bin/bash
for TASK in cola mrpc stsb rte sst2 qnli qqp mnli
do
for LORA in "--lora" ""
do
for SEED in 1 2 3 4 5
do
for RANK in 1 2 4 8 16
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --seed ${SEED} --rank ${RANK} --v 0.1"
rm -f ${TASK}_${LORA}_${SEED}_rank${RANK}_output.log ${TASK}_${LORA}_${SEED}_rank${RANK}_error.log
python3 spikelora_finetuning/deberta_chpc.py --task "$TASK" ${LORA} --seed ${SEED} --rank ${RANK} --v 0.1 > ${TASK}_${LORA}_${SEED}_rank${RANK}_output.log 2> ${TASK}_${LORA}_${SEED}_rank${RANK}_error.log
done
done