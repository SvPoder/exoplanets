import sys
sys.path.append("/home/mariacst/cluster/.env/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata
from utils import heat, temperature_withDM
from mock_generation import mock_population
from astropy.constants import R_jup, M_sun
import glob
import pickle
import time

start = time.time()

# Local DM density
rho0 = 0.42 # GeV/cm3

rank          = 1.
nBDs          = int(sys.argv[1])
rel_unc_Tobs  = 0.1
rel_mass      = 0.1
f_true        = 1.
gamma_true    = float(sys.argv[2])
rs_true       = float(sys.argv[3])

# --------- MCMC fitting -> change to another file -----------------------
def lnprior(p):
    f, gamma, rs = p
    if ( 0. < gamma < 3. and 0. < f < 1. and 0.01 < rs < 50.):
        return 0.
    return -np.inf

def residual(p):#, robs, Tobs, rel_unc_Tobs, heat_int, mass):
    f, gamma, rs = p
    Tmodel = temperature_withDM(robs, heat_int, f=f, M=mass*M_sun.value, 
                                parameters=[gamma, rs, rho0])
    return -0.5*np.sum(((Tmodel-Tobs)/(rel_unc_Tobs*Tobs))**2.)


def lnprob(p):#, robs, Tobs, rel_unc_Tobs, heat_int, mass):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p)#, robs, Tobs, rel_unc_Tobs, heat_int, mass)
# ------------------------------------------------------------------------

# Load theoretical cooling model
path = "./data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]

np.random.seed(42)
robs, Tobs, mass, ages = mock_population(nBDs, rel_unc_Tobs, rel_mass,
                                             f_true, gamma_true,
                                             rs_true, rho0_true=rho0)
print(Tobs[0], Tobs[467], robs[34])

## calculate predictic intrinsic heat flow for mock BDs
xi = np.transpose(np.asarray([ages, mass]))
Teff     = griddata(points, values, xi)
heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)
# ------------------ RECONSTRUCTION --------------------------------------
ndim     = 3
nwalkers = 50
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) 
                                #args=(r_obs, Tobs, rel_unc_T, heat_int, mass))
pos, prob, state  = sampler.run_mcmc(p0, 300, progress=True)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 5000, progress=True)
like    = sampler.flatlnprobability # likelihood
maxlike = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
print ("ML estimator : " , maxlike)

# Save likelihood
#file_object = open(("likelihood_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
#                    %(nBDs, rel_unc_Tobs, rel_mass, f_true, gamma_true, rs_true))
#                    + "v" + str(rank), "wb")
#pickle.dump(like, file_object, protocol=2)
#file_object.close()
# Save posterior
file_object = open(("posterior_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                    %(nBDs, rel_unc_Tobs, rel_mass, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(sampler.flatchain, file_object, protocol=2)
file_object.close()


print("Running time in s =", time.time()-start)
