qsub -v TASK=cola,LORA=true deberta_4.sh -N cola-lora
qsub -v TASK=mrpc,LORA=true deberta_4.sh -N mrpc-lora
qsub -v TASK=stsb,LORA=true deberta_4.sh -N stsb-lora
qsub -v TASK=rte,LORA=true deberta_4.sh -N rte-lora
qsub -v TASK=sst2,LORA=true deberta_12.sh -N sst2-lora
qsub -v TASK=qnli,LORA=true deberta_12.sh -N qnli-lora
qsub -v TASK=qqp,LORA=true deberta_12.sh -N qqp-lora
qsub -v TASK=mnli,LORA=true deberta_12.sh -N mnli-lora