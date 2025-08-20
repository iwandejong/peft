#!/bin/bash
#PBS -N spikelora-pyhopper
#PBS -q gpu_1
#PBS -l select=1:ncpus=10:ngpus=1
#PBS -P CSCI1166
#PBS -l walltime=12:00:00
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

# if [[ $HOSTNAME == gpu200* ]]; then
#   module load chpc/cuda/11.6/PCIe/11.6
# elif [[ $HOSTNAME == gpu400* ]]; then
#   module load chpc/cuda/11.6/SXM2/11.6
# else
#   echo "Unknown GPU node type: $HOSTNAME"
#   exit 1
# fi

module load chpc/python/anaconda/3-2021.11

# Activate venv
source /mnt/lustre/users/idejong/venv/bin/activate
export LD_LIBRARY_PATH=/mnt/lustre/users/idejong/peft/venv/lib:$LD_LIBRARY_PATH

# Check CUDA version and availability
python3 -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"

TASK=${TASK:-sst2}
 
# run 
rm spikelora_output.log spikelora_error.log
python3 examples/spikelora_finetuning/deberta.py --task="$TASK" > ${TASK}_output.log 2> ${TASK}_error.log