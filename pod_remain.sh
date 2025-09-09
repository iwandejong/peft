#!/bin/bash
# MNLI, seed 5, spikelora
python3 spikelora_finetuning/deberta_chpc.py --task mnli --seed 5 > r_mnli_5_output.log 2> r_mnli_5_error.log

# QQP, seed 5, lora
python3 spikelora_finetuning/deberta_chpc.py --task qqp --seed 5 --lora > r_qqp_5_output.log 2> r_qqp_5_error.log

# remove runpod
sleep 60
runpodctl remove pod $RUNPOD_POD_ID