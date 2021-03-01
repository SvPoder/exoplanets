import emcee
import numpy as np
from scipy.interpolate import interp1d, griddata
from utils import heat, temperature_withDM
from mock_generation import mock_population
from astropy.constants import R_jup, M_sun
import glob
import sys
import pickle
from mpi4py import MPI

# Local DM density
rho0 = 0.42 # GeV/cm3

nBDs       = int(sys.argv[1])
rel_unc_T  = float(sys.argv[2])
rel_mass   = float(sys.argv[3])
f_true     = float(sys.argv[4])
gamma_true = float(sys.argv[5])
rs_true    = 20.

# --------- MCMC fitting -> change to another file -----------------------
def lnprior(p):
    f, gamma, rs = p
    if ( 0. < gamma < 2. and 0. < f < 1. and 0.01 < rs < 50.):
        return 0.
    return -np.inf

def residual(p, robs, Tobs, rel_unc_Tobs, heat_int, mass):
    f, gamma, rs = p
    Tmodel = temperature_withDM(robs, heat_int, f=f, M=mass*M_sun.value, 
                                parameters=[gamma, rs, rho0])
    return -0.5*np.sum(((Tmodel-Tobs)/(rel_unc_Tobs*Tobs))**2.)


def lnprob(p, robs, Tobs, rel_unc_Tobs, heat_int, mass):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p, robs, Tobs, rel_unc_Tobs, heat_int, mass)
# ------------------------------------------------------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ------------------ SIMULATION ------------------------------------------
## mock sample of BDs
r_obs, Tobs, mass, ages = mock_population(nBDs, rel_unc_T, rel_mass,
                                          f_true, gamma_true, 
                                          rs_true=rs_true, rho0_true=rho0)
# ------------------ RECONSTRUCTION --------------------------------------
## calculate predictic intrinsic heat flow for mock BDs
#heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)
path = "./data/"
M     = []
age   = {}
Teff  = {}
files = glob.glob(path + "*.txt")
for file in files:
    data = np.genfromtxt(file, unpack=True)
    age[data[0][0]]  = data[1] # age [Gyr]
    Teff[data[0][0]] = data[2] # Teff [K]
    M.append(data[0][0])

_age   = np.linspace(1, 10, 100)
_age_i = []; _mass = []; _teff = []
# the first 5 masses do not have all values between 1 and 10 Gyr
M = np.sort(M)[5:-10] # further remove larger masses
for m in M:
    Teff_interp = interp1d(age[m], Teff[m])
    for _a in _age:
        _age_i.append(_a)
        _mass.append(m)
        _teff.append(Teff_interp(_a))
points = np.transpose(np.asarray([_age_i, _mass]))
values = np.asarray(_teff)
xi = np.transpose(np.asarray([ages, mass]))
Teff     = griddata(points, values, xi)
heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)

ndim     = 3
nwalkers = 50
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(r_obs, Tobs, rel_unc_T, heat_int, mass))
pos, prob, state  = sampler.run_mcmc(p0, 300, progress=False)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 5000, progress=False)
like    = sampler.flatlnprobability # likelihood
samples = sampler.flatchain # posterior
maxlike = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
print ("ML estimator : " , maxlike)

# Save likelihood
file_object = open("./results/likelihood_" + 
                   ("ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                    %(nBDs, rel_unc_T, rel_mass, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(like, file_object, protocol=2)
file_object.close()

# Save posterior
file_object = open("./results/posterior_" + 
                   ("ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                    %(nBDs, rel_unc_T, rel_mass, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(samples, file_object, protocol=2)
file_object.close()


