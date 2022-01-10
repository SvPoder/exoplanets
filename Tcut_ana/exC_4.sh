#!/bin/bash
#SBATCH --array=1-100

source /home/mariacst/exoplanets/running/.env/bin/activate

ex=fixedT10v100Tcut650_nocutTwn
N=100
sigma=0.1

python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.6 5. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.7 5. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.8 5. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.9 5. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.6 10. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.7 10. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.8 10. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.9 10. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.6 20. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.7 20. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.8 20. $SLURM_JOB_ID
python3.6 fitting_Ntimesb.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.9 20. $SLURM_JOB_ID
