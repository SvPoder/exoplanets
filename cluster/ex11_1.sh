#!/bin/bash
#SBATCH --array=1-100

source /home/mariacst/cluster/.env/bin/activate

ex=ex11
N=1000
relT=0.3
relM=0.3
relR=0.3
relA=0.3

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 0. 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 0.5 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1. 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1.1 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1.2 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1.3 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1.4 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1.5 10.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 0. 20.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 0.5 20.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1. 20.
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $relT $relM $relR $relA 1.1 20.
