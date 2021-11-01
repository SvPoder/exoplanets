#!/bin/bash
#set -x #print out all commands before executing
#set -e #abort bash script on error
#SBATCH --job-name=BD_Reconstruct
#SBATCH --time=00-00:30:00
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -e /home/sven/repos/exoplanets/logs/error_BD100_%A_%a.log
#SBATCH -o /home/sven/repos/exoplanets/logs/output_BD100_%A_%a.log
#SBATCH --array=1-3

source /home/sven/exoplanetenv/bin/activate

env

ex=fixedT10v100_test
N=100
sigma=0.1

python3.6 fitting_Ntimes_100.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.5 5.

echo "Finished script 1."

python3.6 fitting_Ntimes_100.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.6 5.

echo "Finished script 2."

sigma=0.2
python3.6 fitting_Ntimes_100.py $ex $SLURM_ARRAY_TASK_ID $N $sigma 0.7 10.

echo "Finished script 3."
