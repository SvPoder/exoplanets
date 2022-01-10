#!/bin/bash
#SBATCH --array=41-100
#SBATCH --partition=short
source /home/mariacst/exoplanets/running/.env/bin/activate

ex=fixedT10v100Tcut650_nocutTwn
N=100
sigma=0.2

python3.6 fitting_Ntimes_check.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.2 20.
