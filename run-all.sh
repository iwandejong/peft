qsub -v TASK=cola deberta_4.sh -N cola-spike
qsub -v TASK=mrpc deberta_4.sh -N mrpc-spike
qsub -v TASK=stsb deberta_4.sh -N stsb-spike
qsub -v TASK=rte deberta_4.sh -N rte-spike
qsub -v TASK=sst2 deberta_12.sh -N sst2-spike
qsub -v TASK=qnli deberta_12.sh -N qnli-spike
qsub -v TASK=qqp deberta_12.sh -N qqp-spike
qsub -v TASK=mnli deberta_12.sh -N mnli-spike