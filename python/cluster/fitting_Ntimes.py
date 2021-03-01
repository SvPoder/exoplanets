import emcee
import numpy as np
from scipy.interpolate import interp1d, interp2d
from utils import temperature, heat, temperature_withDM
from mock_generation import mock_population
from astropy.constants import R_jup, M_jup, M_sun, L_sun
import pickle
import sys
from mpi4py import MPI

# Local DM density
rho0 = 0.42 # GeV/cm3

nBDs         = int(sys.argv[1])
rel_unc_Tobs = float(sys.argv[2])
rel_mass     = float(sys.argv[3])
f_true       = float(sys.argv[4])
gamma_true   = float(sys.argv[5])
rs_true      = 20.

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

if rank==0:
    ## load theoretical BD cooling model taken from Saumon & Marley '08 (fig 2)
    age  = {}; logL = {}; L = {}; Teff = {}
    M    = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    # TODO simplify by directly interpolating heating/luminosity
    for m in M:
        data = np.genfromtxt("./data/saumon_marley_fig2_" + str(m) + ".dat", 
                         unpack=True)
        age[m]  = data[0]
        heat_int   = np.power(10, data[1])*L_sun.value
        Teff[m] = temperature(heat_int, R_jup)
    log_age  = np.linspace(6.1, 9.92, 10)
    _log_age = []
    _mass    = []
    _teff    = []
    for m in M:
        Teff_interp = interp1d(age[m], Teff[m])
        for lage in log_age:
            _log_age.append(lage)
            _mass.append(m)
            _teff.append(Teff_interp(lage))
    # effective temperature (wo DM heating) vs log(age) and mass exoplanet
    Teff_interp_2d = interp2d(_log_age, _mass, _teff)
else:
    Teff_interp_2d=None
Teff_interp_2d = comm.bcast(Teff_interp_2d, root=0)

## mock sample of BDs
r_obs, Tobs, rel_unc_Tobs, mass, log_ages = mock_population(nBDs, rel_unc_Tobs,
                                                            rel_mass,
                                                            f_true, gamma_true, 
                                                            rs_true=rs_true)

## calculate predictic intrinsic heat flow for mock BDs
heat_int = np.zeros(len(r_obs))
for i in range(len(r_obs)):
    heat_int[i] = heat(Teff_interp_2d(log_ages[i], mass[i]), R_jup.value)

print("Relative uncertainty in Tobs is ", rel_unc_Tobs)

ndim     = 3
nwalkers = 50
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(r_obs, Tobs, rel_unc_Tobs, heat_int, mass))
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
                    %(nBDs, rel_unc_Tobs, rel_mass, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(like, file_object, protocol=2)
file_object.close()

# Save posterior
file_object = open("./results/posterior_" + 
                   ("ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                    %(nBDs, rel_unc_Tobs, rel_mass, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(samples, file_object, protocol=2)
file_object.close()


