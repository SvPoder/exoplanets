import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata
from derivatives import derivativeTint_wrt_A, derivativeTint_wrt_M
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
from astropy.constants import R_jup
import glob
import pickle
#import time
from emcee_functions import lnprob

#start = time.time()

# Constant parameters & conversions ==========================================
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
# ============================================================================
# Input parameters
ex         = sys.argv[1]
rank       = int(sys.argv[2])
nBDs       = int(sys.argv[3])
relTobs    = 0.1
sigma      = float(sys.argv[4])
f_true     = 1.
gamma_true = float(sys.argv[5])
rs_true    = float(sys.argv[6])
v          = 100.
# ------------------------------------------------------------------------
# Load theoretical cooling model
path = "/home/mariacst/exoplanets/running/data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]
# Mock observation
#np.random.seed(rank)
(robs, sigmarobs, Tobs, sigmaTobs, Mobs,
     sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relTobs, sigma,
                                      sigma, sigma, f_true, gamma_true,
                                      rs_true, rho0_true=rho0, v=v)
# Calculate derivatives Tint wrt Age and Mass                               
dervTint_A = np.ones(nBDs)                                                  
dervTint_M = np.ones(nBDs)                                                  
size       = 7000                                                           
h          = 0.001                                                          
for i in range(nBDs):                                                       
    dervTint_A[i] = derivativeTint_wrt_A(Mobs[i], Aobs[i], points, values,  
                                         size=size, h=h)                        
    dervTint_M[i] = derivativeTint_wrt_M(Mobs[i], Aobs[i], points, values,  
                                         size=size, h=h)  

## calculate predictic intrinsic temperature
xi       = np.transpose(np.asarray([Aobs, Mobs]))
Teff     = griddata(points, values, xi)
# ------------------ RECONSTRUCTION --------------------------------------
ndim     = 3
nwalkers = 150
print(nwalkers)
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
              args=(robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs,
                    sigmaTobs, Teff, points, values, dervTint_M, dervTint_A,
                    v, R_jup.value, Rsun, rho0, epsilon))

pos, prob, state  = sampler.run_mcmc(p0, 200, progress=False)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 6000, progress=False)

# Save likelihood
_path = "/hdfs/local/mariacst/exoplanets/results/likelihood/velocity/v100/fixedT10/"
filepath    = (_path + "N%isigma%.1fb/like_" %(nBDs, sigma) + ex)
file_object = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))            
                    + "v" + str(rank), "wb") 
pickle.dump(sampler.flatlnprobability, file_object, protocol=2)
file_object.close() 
# Save posterior
_path = "/hdfs/local/mariacst/exoplanets/results/posterior/velocity/v100/fixedT10/"
filepath    = (_path + "N%isigma%.1fb/posterior_" %(nBDs, sigma) + ex)       
file_object2 = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))            
                    + "v" + str(rank), "wb")                                  
pickle.dump(sampler.flatchain, file_object2, protocol=2)
file_object2.close()
