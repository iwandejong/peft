qsub -v TASK=cola,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N cola-lora
qsub -v TASK=mrpc,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N mrpc-lora
qsub -v TASK=stsb,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N stsb-lora
qsub -v TASK=rte,LORA=true,WANDB_PROJECT=chpc deberta_4.sh -N rte-lora
qsub -v TASK=sst2,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N sst2-lora
qsub -v TASK=qnli,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N qnli-lora
qsub -v TASK=qqp,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N qqp-lora
qsub -v TASK=mnli,LORA=true,WANDB_PROJECT=chpc deberta_12.sh -N mnli-lora