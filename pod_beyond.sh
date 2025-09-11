#!/bin/bash
# beyond rank
for TASK in cola
do
for LORA in "" "--lora"
do
for RANK in 64
do
for LR in 5e-4
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project chpc --lr $LR"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project chpc --lr $LR > logs/${TASK}_r${RANK}_lr${LR}_${LORA//--/--}.log 2>&1
done
done
done
done

# beyond learning rate
for TASK in cola
do
for LORA in "" "--lora"
do
for LR in 1.1e-3
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project lrs --lr $LR"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project lrs --lr $LR > logs/${TASK}_r${RANK}_lr${LR}_${LORA//--/--}.log 2>&1
done
done
done

# performance
for TASK in cola
do
for LORA in "" "--lora"
do
echo "Running: python3 spikelora_finetuning/performance.py --task $TASK --project performance $LORA"
python3 spikelora_finetuning/performance.py --task $TASK --project performance $LORA > logs/performance_${TASK}_${LORA//--/--}.log 2>&1
done
done