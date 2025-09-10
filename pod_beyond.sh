#!/bin/bash
for TASK in rte
do
for LORA in "--lora"
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK $LORA > logs/${TASK}_${LORA//--/--}.log 2>&1
done
done

for TASK in cola
do
for LORA in "" "--lora"
do
for RANK in 64 128
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --rank $RANK $LORA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --rank $RANK $LORA > logs/${TASK}_r${RANK}${LORA//--/--}.log 2>&1
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

sleep 60
runpodctl remove pod $RUNPOD_POD_ID