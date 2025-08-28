qsub -v TASK=cola deberta_runs.sh -N cola-spike
qsub -v TASK=mrpc deberta_runs.sh -N mrpc-spike
qsub -v TASK=stsb deberta_runs.sh -N stsb-spike
qsub -v TASK=rte deberta_runs.sh -N rte-spike
qsub -v TASK=sst2 deberta_12_runs.sh -N sst2-spike
qsub -v TASK=qnli deberta_12_runs.sh -N qnli-spike
qsub -v TASK=qqp deberta_12_runs.sh -N qqp-spike
qsub -v TASK=mnli deberta_12_runs.sh -N mnli-spike