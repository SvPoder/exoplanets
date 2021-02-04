#!/bin/bash
#SBATCH -t 24:00:00 --mem 16G
#SBATCH -n 100

module load anaconda3
module load intel/compilers
module load intel/mpi

goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.1 0.2
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.1 0.6
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.1 1.0
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.1 1.4
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.1 1.8
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.3 0.2
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.3 0.6
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.3 1.0
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.3 1.4
goo-job-nanny srun -n 100 python3.6 fitting_Ntimes.py 100 0.05 0.3 1.8
