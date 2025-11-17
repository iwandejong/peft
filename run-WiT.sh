qsub -v TASK=cola,SEED=1,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-1-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=2,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-2-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=3,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-3-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=4,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-4-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=5,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-5-1 -l walltime=01:00:00 #

qsub -v TASK=cola,SEED=2,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-2-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=1,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-1-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=3,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-3-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=4,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-4-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=5,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-5-2 -l walltime=01:00:00 #

qsub -v TASK=cola,SEED=1,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-1-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-2-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-3-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-4-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-5-4 -l walltime=01:30:00

qsub -v TASK=cola,SEED=1,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-1-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-2-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-3-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-4-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-5-16 -l walltime=01:30:00

qsub -v TASK=cola,SEED=1,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-1-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-2-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-3-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-4-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N cola-spike-5-32 -l walltime=01:30:00

# lr
qsub -v TASK=cola,SEED=1,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N cola-spike-1-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=2,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N cola-spike-2-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=3,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N cola-spike-3-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=4,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N cola-spike-4-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=5,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N cola-spike-5-1 -l walltime=01:00:00

qsub -v TASK=cola,SEED=2,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N cola-spike-2-2 -l walltime=01:00:00
qsub -v TASK=cola,SEED=1,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N cola-spike-1-2 -l walltime=01:00:00
qsub -v TASK=cola,SEED=3,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N cola-spike-3-2 -l walltime=01:00:00
qsub -v TASK=cola,SEED=4,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N cola-spike-4-2 -l walltime=01:00:00
qsub -v TASK=cola,SEED=5,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N cola-spike-5-2 -l walltime=01:00:00

qsub -v TASK=cola,SEED=1,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N cola-spike-1-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N cola-spike-2-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N cola-spike-3-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N cola-spike-4-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N cola-spike-5-4 -l walltime=01:30:00

qsub -v TASK=cola,SEED=1,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N cola-spike-1-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N cola-spike-2-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N cola-spike-3-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N cola-spike-4-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N cola-spike-5-4 -l walltime=01:30:00
