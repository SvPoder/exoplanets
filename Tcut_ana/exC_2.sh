#!/bin/bash
#SBATCH --array=1-100

source /home/mariacst/exoplanets/running/.env/bin/activate

ex=fixedT10v100Tcut650_nocutTwn
N=1000
sigma=0.1

python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.5 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1. 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.1 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.2 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.3 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.4 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.5 5. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.5 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1. 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.1 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.2 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.3 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.4 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.5 10. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.5 20. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1. 20. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.1 20. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.2 20. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.3 20. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.4 20. $SLURM_JOB_ID
python3.6 fitting_Ntimes.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 1.5 20. $SLURM_JOB_ID
