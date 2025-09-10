#!/bin/bash
for TASK in cola
do
for LORA in "" "--lora"
do
for RANK in 64 128
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA > logs/${TASK}_r${RANK}${LORA//--/--}.log 2>&1
done
done
done

for TASK in cola
do
for LORA in "" "--lora"
do
for LR in 1.1e-3 1.3e-3
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --lr $LR $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --lr $LR $LORA > logs/${TASK}_lr${LR}${LORA//--/--}.log 2>&1
done
done
done

sleep 60
runpodctl remove pod $RUNPOD_POD_ID