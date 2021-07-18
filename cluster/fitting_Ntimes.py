import sys
sys.path.append("/home/mariacst/cluster/.env/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata
from utils import heat, temperature_withDM
from mock_generation import mock_population_all
from astropy.constants import R_jup, M_sun
import glob
import pickle
import time

start = time.time()

# Local DM density
rho0 = 0.42 # GeV/cm3

ex         = sys.argv[1]
rank       = int(sys.argv[2])
nBDs       = int(sys.argv[3])
relTobs    = float(sys.argv[4])
relM       = float(sys.argv[5])
relRobs    = float(sys.argv[6])
relA       = float(sys.argv[7])
f_true     = 1.
gamma_true = float(sys.argv[8])
rs_true    = float(sys.argv[9])
try:
    v = float(sys.argv[8])
except:
    v = None
#print(v)
# --------- MCMC fitting -> change to another file -----------------------
def lnprior(p):
    f, gamma, rs = p
    if ( 0. < gamma < 3. and 0. < f < 1. and 0.01 < rs < 50.):
        return 0.
    return -np.inf

def residual(p):#, robs, Tobs, rel_unc_Tobs, heat_int, mass):
    f, gamma, rs = p
    Tmodel = temperature_withDM(robs, heat_int, f=f, M=mass*M_sun.value, 
                                parameters=[gamma, rs, rho0], v=v)
    return -0.5*np.sum(((Tmodel-Tobs)/(relTobs*Tobs))**2.)

def lnprob(p):#, robs, Tobs, rel_unc_Tobs, heat_int, mass):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p)#, robs, Tobs, rel_unc_Tobs, heat_int, mass)
# ------------------------------------------------------------------------
# Load theoretical cooling model
path = "/home/mariacst/exoplanets/running/data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]
# Generate mock observation
robs, Tobs, mass, ages = mock_population_all(nBDs, 
                                         relTobs, relM, relRobs, relA,
                                         f_true, gamma_true, rs_true, 
                                         rho0_true=rho0, 
                                         v=v)
## calculate predictic intrinsic heat flow for mock BDs
xi       = np.transpose(np.asarray([ages, mass]))
Teff     = griddata(points, values, xi)
heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)
# ------------------ RECONSTRUCTION --------------------------------------
ndim     = 3
nwalkers = 50
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) 
                                #args=(r_obs, Tobs, rel_unc_T, heat_int, mass))
pos, prob, state  = sampler.run_mcmc(p0, 300, progress=False)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 5000, progress=False)
# Save posterior
filepath    = ("./results/N%irelT%.2frelM%.2f/posterior_" 
               %(nBDs, relTobs, relM) + ex)
file_object = open(filepath + 
                   ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                    %(nBDs, relTobs, relM, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(sampler.flatchain, file_object, protocol=2)
file_object.close()


print("Running time in s =", time.time()-start)
