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
from astropy.constants import R_jup, M_jup, G, sigma_sb
import glob
import pickle
import time

#from emcee_functions import lnprob

# Timing the script
start = time.time()

# Constant parameters & conversions ==========================================
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
_sigma_sb = sigma_sb.value
_G        = G.value
# ============================================================================
# Input parameters
ex         = "test"
rank       = 1
nBDs       = 100
relTobs    = 0.1
sigma      = 0.1
f_true     = 1.
gamma_true = 1.
rs_true    = 15.
v          = 100.
# ------------------------------------------------------------------------
# Load theoretical cooling model
path = "/home/mariacst/exoplanets/running/data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]

# Mock observation
np.random.seed(36)
(robs, sigmarobs, Tobs, sigmaTobs, Mobs,
     sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relTobs, sigma,
                                      sigma, sigma, f_true, gamma_true,
                                      rs_true, rho0_true=rho0, v=v)

print("Mock")
print(robs[0])