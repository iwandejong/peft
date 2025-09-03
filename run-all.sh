qsub -v TASK=cola,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N cola-lora
qsub -v TASK=mrpc,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N mrpc-lora
qsub -v TASK=stsb,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N stsb-lora
qsub -v TASK=rte,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N rte-lora
qsub -v TASK=sst2,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N sst2-lora
qsub -v TASK=qnli,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N qnli-lora
qsub -v TASK=qqp,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N qqp-lora
qsub -v TASK=mnli,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N mnli-lora

qsub -v TASK=cola,LORA=true,SEED=1,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,LORA=true,SEED=2,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,LORA=true,SEED=3,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,LORA=true,SEED=4,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,LORA=true,SEED=5,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,SEED=1,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,SEED=2,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,SEED=3,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,SEED=4,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,SEED=5,RANK=1 deberta_4.sh -N cola-lora -l walltime=01:00:00

# sst2 (both lora=true and false)
qsub -v TASK=sst2,LORA=true,SEED=1 deberta_4.sh -N sst2-lora -l walltime=4:00:00
qsub -v TASK=sst2,LORA=true,SEED=2 deberta_4.sh -N sst2-lora -l walltime=4:00:00
qsub -v TASK=sst2,LORA=true,SEED=3 deberta_4.sh -N sst2-lora -l walltime=4:00:00
qsub -v TASK=sst2,LORA=true,SEED=4 deberta_4.sh -N sst2-lora -l walltime=4:00:00
qsub -v TASK=sst2,LORA=true,SEED=5 deberta_4.sh -N sst2-lora -l walltime=4:00:00
qsub -v TASK=sst2,SEED=1 deberta_4.sh -N sst2-spike -l walltime=4:00:00
qsub -v TASK=sst2,SEED=2 deberta_4.sh -N sst2-spike -l walltime=4:00:00
qsub -v TASK=sst2,SEED=3 deberta_4.sh -N sst2-spike -l walltime=4:00:00
qsub -v TASK=sst2,SEED=4 deberta_4.sh -N sst2-spike -l walltime=4:00:00
qsub -v TASK=sst2,SEED=5 deberta_4.sh -N sst2-spike -l walltime=4:00:00

qsub -v TASK=cola,SEED=1 deberta_4.sh -N cola-lora -l walltime=01:00:00
qsub -v TASK=cola,SEED=3 deberta_4.sh -N cola-lora -l walltime=01:00:00





