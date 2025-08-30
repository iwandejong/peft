#!/bin/bash
for TASK in cola sst2
do
for SEED in 1 2 3 4 5
do
for V in 0. .01 .05 .1 .25 .5 .75 1.
do
echo "Running: python3 examples/spikelora_finetuning/deberta_chpc_runs.py --task $TASK $LORA --seed $SEED --v $V"
rm -f ${TASK}_${LORA}_${SEED}_v${V}_output.log ${TASK}_${LORA}_${SEED}_v${V}_error.log
python3 spikelora_finetuning/deberta_local.py --task "$TASK" ${LORA} --seed ${SEED} --v ${V} > ${TASK}_${LORA}_${SEED}_v${V}_output.log 2> ${TASK}_${LORA}_${SEED}_v${V}_error.log
done
done