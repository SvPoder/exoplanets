#!/bin/bash
#SBATCH -t 24:00:00 --mem 16G
#SBATCH -n 100

module load anaconda3
module load intel/compilers
module load intel/mpi

goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.01
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.02
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.03
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.05
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.10
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.25
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 0.50
