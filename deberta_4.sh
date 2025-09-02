#!/bin/bash

#PBS -q gpu_1
#PBS -l select=1:ncpus=9:ngpus=1
#PBS -P CSCI1166
#PBS -l walltime=04:00:00
#PBS -m abe
#PBS -M u22498037@tuks.co.za
 
cd /mnt/lustre/users/idejong/peft

export WANDB_API_KEY="ed0a79eefd77fefe9ff353ac45b85404dc20c756"
 
echo
echo `date`: executing CUDA job on host ${HOSTNAME}
echo
echo Available GPU devices: $CUDA_VISIBLE_DEVICES
echo

module purge
module load chpc/python/anaconda/3-2021.11
module load chpc/cuda/12.0/12.0

# Activate venv
source /mnt/lustre/users/idejong/peft/venv/bin/activate

# Check CUDA version and availability
python3 -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

TASK=${TASK:-sst2}
LORA=${LORA:-""}
SEED=${SEED:-0}
if [ "$LORA" = "true" ] ; then
    LORA="--lora"
else
    LORA=""
fi
WANDB_PROJECT=${WANDB_PROJECT:-"deberta-spikelora"}
 
# run 
echo "Running: python3 examples/spikelora_finetuning/deberta_chpc_runs.py --task $TASK $LORA --seed $SEED --wandb_project ${WANDB_PROJECT}"
rm -f ${TASK}_${LORA}_${SEED}_output.log ${TASK}_${LORA}_${SEED}_error.log
python3 spikelora_finetuning/deberta_chpc.py --task "$TASK" ${LORA} --seed ${SEED} --wandb_project "${WANDB_PROJECT}" > ${TASK}_${LORA}_${SEED}_output.log 2> ${TASK}_${LORA}_${SEED}_error.log