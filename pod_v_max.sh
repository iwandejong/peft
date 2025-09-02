#!/bin/bash
for TASK in cola
do
for LORA in "--lora" ""
do
for SEED in 1 2 3 4 5
do
for V in 1.5
do
if [ "$LORA" = "--lora" ]; then
    echo "Using LoRA"
    V=-1.0
else
    echo "Not using LoRA"
fi
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --seed ${SEED} --v ${V} $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task "$TASK" --seed ${SEED} --v ${V} ${LORA} > ${TASK}${LORA}_${SEED}_v${V}_output.log 2> ${TASK}${LORA}_${SEED}_v${V}_error.log
done
done
done
done