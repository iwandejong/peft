#!/bin/bash
for TASK in cola
do
echo "Running: python3 spikelora_finetuning/deberta_chpc.py --task $TASK --project B-BA"
python3 spikelora_finetuning/deberta_chpc.py --task $TASK --project B-BA > logs/${TASK}_BA.log 2>&1
done

sleep 60
runpodctl remove pod $RUNPOD_POD_ID