qsub -v TASK=cola,SEED=1,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N WiT-1-r-1 -l walltime=00:05:00 #

qsub -v TASK=cola,SEED=1,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N WiT-1-r-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=2,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N WiT-2-r-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=3,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N WiT-3-r-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=4,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N WiT-4-r-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=5,RANK=1,PROJECT=WiT-ranks deberta_4.sh -N WiT-5-r-1 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=2,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N WiT-2-r-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=1,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N WiT-1-r-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=3,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N WiT-3-r-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=4,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N WiT-4-r-2 -l walltime=01:00:00 #
qsub -v TASK=cola,SEED=5,RANK=2,PROJECT=WiT-ranks deberta_4.sh -N WiT-5-r-2 -l walltime=01:00:00
qsub -v TASK=cola,SEED=1,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N WiT-1-r-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N WiT-2-r-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N WiT-3-r-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N WiT-4-r-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=4,PROJECT=WiT-ranks deberta_4.sh -N WiT-5-r-4 -l walltime=01:30:00
qsub -v TASK=cola,SEED=1,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N WiT-1-r-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N WiT-2-r-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N WiT-3-r-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N WiT-4-r-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=16,PROJECT=WiT-ranks deberta_4.sh -N WiT-5-r-16 -l walltime=01:30:00
qsub -v TASK=cola,SEED=1,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N WiT-1-r-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N WiT-2-r-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N WiT-3-r-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N WiT-4-r-32 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,RANK=32,PROJECT=WiT-ranks deberta_4.sh -N WiT-5-r-32 -l walltime=01:30:00

# lr
qsub -v TASK=cola,SEED=1,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N WiT-1-lr-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=2,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N WiT-2-lr-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=3,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N WiT-3-lr-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=4,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N WiT-4-lr-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=5,LR=0.0001,PROJECT=WiT-lr deberta_4.sh -N WiT-5-lr-1 -l walltime=01:00:00
qsub -v TASK=cola,SEED=2,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N WiT-2-lr-5 -l walltime=01:00:00
qsub -v TASK=cola,SEED=1,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N WiT-1-lr-5 -l walltime=01:00:00
qsub -v TASK=cola,SEED=3,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N WiT-3-lr-5 -l walltime=01:00:00
qsub -v TASK=cola,SEED=4,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N WiT-4-lr-5 -l walltime=01:00:00
qsub -v TASK=cola,SEED=5,LR=0.0005,PROJECT=WiT-lr deberta_4.sh -N WiT-5-lr-5 -l walltime=01:00:00
qsub -v TASK=cola,SEED=1,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N WiT-1-lr-7 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N WiT-2-lr-7 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N WiT-3-lr-7 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N WiT-4-lr-7 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,LR=0.0007,PROJECT=WiT-lr deberta_4.sh -N WiT-5-lr-7 -l walltime=01:30:00
qsub -v TASK=cola,SEED=1,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N WiT-1-lr-9 -l walltime=01:30:00
qsub -v TASK=cola,SEED=2,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N WiT-2-lr-9 -l walltime=01:30:00
qsub -v TASK=cola,SEED=3,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N WiT-3-lr-9 -l walltime=01:30:00
qsub -v TASK=cola,SEED=4,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N WiT-4-lr-9 -l walltime=01:30:00
qsub -v TASK=cola,SEED=5,LR=0.0009,PROJECT=WiT-lr deberta_4.sh -N WiT-5-lr-9 -l walltime=01:30:00
