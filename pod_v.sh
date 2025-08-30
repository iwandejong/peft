#!/bin/bash
for TASK in cola sst2
do
for SEED in 1 2 3 4 5
do
for V in 0. .01 .05 .1 .25 .5 .75 1.
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --seed ${SEED} --v ${V}"
rm -f ${TASK}_${SEED}_v${V}_output.log ${TASK}_${SEED}_v${V}_error.log
python3 spikelora_finetuning/deberta_chpc.py --task "$TASK" --seed ${SEED} --v ${V} > ${TASK}_${SEED}_v${V}_output.log 2> ${TASK}_${SEED}_v${V}_error.log
done
done
done