qsub -v TASK=cola,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N cola-lora
qsub -v TASK=mrpc,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N mrpc-lora
qsub -v TASK=stsb,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N stsb-lora
qsub -v TASK=rte,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N rte-lora
qsub -v TASK=sst2,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N sst2-lora
qsub -v TASK=qnli,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N qnli-lora
qsub -v TASK=qqp,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N qqp-lora
qsub -v TASK=mnli,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N mnli-lora

# qsub -v TASK=cola,LORA=true,SEED=1,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=2,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=3,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=4,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=5,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=1,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=2,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=3,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=4,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=5,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00 #

# qsub -v TASK=cola,LORA=true,SEED=1,RANK=2 deberta_4.sh -N cola-lora-1-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=2,RANK=2 deberta_4.sh -N cola-lora-2-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=3,RANK=2 deberta_4.sh -N cola-lora-3-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=4,RANK=2 deberta_4.sh -N cola-lora-4-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=5,RANK=2 deberta_4.sh -N cola-lora-5-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=2,RANK=2 deberta_4.sh -N cola-spike-2-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=1,RANK=2 deberta_4.sh -N cola-spike-1-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=3,RANK=2 deberta_4.sh -N cola-spike-3-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=4,RANK=2 deberta_4.sh -N cola-spike-4-2 -l walltime=01:00:00 #
# qsub -v TASK=cola,SEED=5,RANK=2 deberta_4.sh -N cola-spike-5-2 -l walltime=01:00:00 #

# qsub -v TASK=cola,LORA=true,SEED=2,RANK=4 deberta_4.sh -N cola-lora-1-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=1,RANK=4 deberta_4.sh -N cola-lora-2-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=3,RANK=4 deberta_4.sh -N cola-lora-3-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=4,RANK=4 deberta_4.sh -N cola-lora-4-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=5,RANK=4 deberta_4.sh -N cola-lora-5-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=1,RANK=4 deberta_4.sh -N cola-spike-1-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=2,RANK=4 deberta_4.sh -N cola-spike-2-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=3,RANK=4 deberta_4.sh -N cola-spike-3-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=4,RANK=4 deberta_4.sh -N cola-spike-4-4 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=5,RANK=4 deberta_4.sh -N cola-spike-5-4 -l walltime=01:30:00 #

# qsub -v TASK=cola,LORA=true,SEED=1,RANK=16 deberta_4.sh -N cola-lora-1-16 -l walltime=03:00:00 #
# qsub -v TASK=cola,LORA=true,SEED=2,RANK=16 deberta_4.sh -N cola-lora-2-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=3,RANK=16 deberta_4.sh -N cola-lora-3-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=4,RANK=16 deberta_4.sh -N cola-lora-4-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=5,RANK=16 deberta_4.sh -N cola-lora-5-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=1,RANK=16 deberta_4.sh -N cola-spike-1-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=2,RANK=16 deberta_4.sh -N cola-spike-2-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=3,RANK=16 deberta_4.sh -N cola-spike-3-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=4,RANK=16 deberta_4.sh -N cola-spike-4-16 -l walltime=01:30:00 #
# qsub -v TASK=cola,SEED=5,RANK=16 deberta_4.sh -N cola-spike-5-16 -l walltime=01:30:00 #

# qsub -v TASK=cola,LORA=true,SEED=1,RANK=32 deberta_4.sh -N cola-lora-1-32 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=2,RANK=32 deberta_4.sh -N cola-lora-2-32 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=3,RANK=32 deberta_4.sh -N cola-lora-3-32 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=4,RANK=32 deberta_4.sh -N cola-lora-4-32 -l walltime=01:30:00 #
# qsub -v TASK=cola,LORA=true,SEED=5,RANK=32 deberta_4.sh -N cola-lora-5-32 -l walltime=01:30:00 #
qsub -v TASK=cola,SEED=1,RANK=32 deberta_4.sh -N cola-spike-1-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=32 deberta_4.sh -N cola-spike-2-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=32 deberta_4.sh -N cola-spike-3-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=32 deberta_4.sh -N cola-spike-4-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=32 deberta_4.sh -N cola-spike-5-32 -l walltime=01:30:00

