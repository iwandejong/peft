#!/bin/bash
# MNLI, seed 5, spikelora
python3 spikelora_finetuning/deberta_chpc.py --task mnli --seed 5 > r:mnli_5_output.log 2> r:mnli_5_error.log

# QQP, seed 5, lora
python3 spikelora_finetuning/deberta_chpc.py --task qqp --seed 5 --lora > r:qqp_5_output.log 2> r:qqp_5_error.log