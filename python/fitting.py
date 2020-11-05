import emcee
import numpy as np
from scipy.interpolate import interp1d, interp2d
from utils import temperature, heat, temperature_withDM
from mock_generation import mock_population
from astropy.constants import R_jup, M_jup, M_sun, L_sun
import pickle

# Local DM density
rho0 = 0.42 # GeV/cm3


## load theoretical BD cooling model taken from Saumon & Marley '08 (fig 2)
age  = {}; logL = {}; L = {}; Teff = {}
M    = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]


# TODO simplify by directly interpolating heating/luminosity
for mass in M:
    data = np.genfromtxt("../data/saumon_marley_fig2_" + str(mass) + ".dat", 
                         unpack=True)
    age[mass]  = data[0]
    heat_int   = np.power(10, data[1])*L_sun.value
    Teff[mass] = temperature(heat_int, R_jup)
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

# TODO think better way of dealing w/ mass and age modelled exoplanets
# Assumption for all exoplanets
L_age    = 9.69
M        = 40.
heat_int = heat(Teff_interp_2d(L_age, M*M_jup/M_sun), R_jup.value)

def lnprior(p):
    gamma, f = p
    if ( 0. < gamma < 2. and 0. < f < 1.):
        return 0.
    return -np.inf

def residual(p, robs, Tobs, rel_unc_Tobs):
    gamma, f = p
    Tmodel = temperature_withDM(robs, heat_int, f=f, M=M*M_jup.value, 
                                parameters=[gamma, 20., rho0])
    return -0.5*np.sum(((Tmodel-Tobs)/(rel_unc_Tobs*Tobs))**2.)


def lnprob(p, robs, Tobs, rel_unc_Tobs):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p, robs, Tobs, rel_unc_Tobs)


r_obs, Tobs, rel_unc_Tobs, Teff, mass, log_ages = mock_population(10000)
# Select a subsample
mass_bins = np.linspace(14, 75, 4)*M_jup/M_sun
age_bins = np.linspace(9, 9.92, 4)

pos = np.where((mass > mass_bins[1]) & (mass < mass_bins[2]) & 
               (log_ages > age_bins[2]) & (log_ages < age_bins[3]))
print(mass_bins[1]*M_sun/M_jup, mass_bins[2]*M_sun/M_jup)
print(10**age_bins[2]/1e9, 10**age_bins[3]/1e9)
print("In subsample there are %i exoplanets" %len(pos[0]))

ndim     = 2
nwalkers = 50
# first guess
p0 = [[0.9, 0.9] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(r_obs[pos], Tobs[pos], rel_unc_Tobs))
pos, prob, state  = sampler.run_mcmc(p0, 300, progress=True)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 5000, progress=True)
like    = sampler.flatlnprobability # likelihood
samples = sampler.flatchain # posterior
maxlike = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
print ("ML estimator : " , maxlike)

# Save likelihood
file_object = open("../results/likelihood_2", "wb")
pickle.dump(like, file_object, protocol=2)
file_object.close()

# Save posterior
file_object = open("../results/posterior_2", "wb")
pickle.dump(samples, file_object, protocol=2)
file_object.close()
