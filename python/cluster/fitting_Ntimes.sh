#!/bin/bash
#SBATCH -t 1:00:00 --mem 16G
#SBATCH -n 20

module load anaconda3
module load intel/compilers
module load intel/mpi

goo-job-nanny srun -n 20 python3.6 fitting_Ntimes.py
