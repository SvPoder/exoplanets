#!/bin/bash

source /home/svenpoder/exoplanetenv/bin/activate

ex=like_post_data_100
N=100

echo "Starting 100 BD runs"
for i in {1..20}
do
  echo "Running $ex for rank $i"
  python3 Tcut_ana/fitting_Ntimes_debug.py $ex $i $N 0.2 1.2 20
done

ex=like_post_data_1000
N=1000

echo "Starting 1000 BD runs"
for i in {1..20}
do
  echo "Running $ex for rank $i"
  python3 Tcut_ana/fitting_Ntimes_debug.py $ex $i $N 0.2 1.2 20
done

# source /home/svenpoder/exoplanetenv/bin/activate

# python3 Tcut_ana/fitting_Ntimes_debug.py TEST_100_sig02_NO_SIGMA_RS_SQUEEZE 1 100 0.2 1.2 20
# python3 Tcut_ana/fitting_Ntimes_debug.py TEST_500_sig02_NO_SIGMA_RS_SQUEEZE 1 500 0.2 1.2 20
# python3 Tcut_ana/fitting_Ntimes_debug.py TEST_1000_sig02_NO_SIGMA_RS_SQUEEZE 1 1000 0.2 1.2 20