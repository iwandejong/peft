#!/bin/bash
# beyond rank
for TASK in cola
do
for LORA in "" "--lora"
do
for RANK in 64
do
for LR in 1e-4
do
for BZ in 8
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project chpc --lr $LR --bz $BZ"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project chpc --lr $LR --bz $BZ > logs/${TASK}_r${RANK}_lr${LR}_bz${BZ}${LORA//--/--}.log 2>&1
done
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
for BZ in 8
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project lrs --lr $LR --bz $BZ"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --r $RANK $LORA --project lrs --lr $LR --bz $BZ > logs/${TASK}_r${RANK}_lr${LR}_bz${BZ}${LORA//--/--}.log 2>&1
done
done
done
done
done