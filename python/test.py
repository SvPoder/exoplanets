from mock_generation import mock_population, mock_population_all
import numpy as np

N=100000; relT=0.1; relM=0.1; f=1; g=0.5; rs=10.
i=286
np.random.seed(42)
r1, T1, m1, a1 = mock_population(N, relT, relM, f, g, rs)
print(r1[i], T1[i], m1[i], a1[i])
