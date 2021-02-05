import emcee
import numpy as np
from utils import temperature, heat, temperature_withDM
from mock_generation import mock_population_old
from astropy.constants import R_jup, M_jup, M_sun, L_sun
import pickle

# Local DM density
rho0 = 0.42 # GeV/cm3


# TODO think better way of dealing w/ mass and age modelled exoplanets
# Assumption for all exoplanets

def lnprior(p):
    gamma, f = p
    if ( 0. < gamma < 2. and 0. < f < 1.):
        return 0.
    return -np.inf

def residual(p, robs, Tobs, rel_unc_Tobs):
    gamma, f = p
    Tmodel = temperature_withDM(robs, 1.1e21, f=f, M=75*M_jup.value, 
                                parameters=[gamma, 20., rho0])
    return -0.5*np.sum(((Tmodel-Tobs)/(rel_unc_Tobs*Tobs))**2.)


def lnprob(p, robs, Tobs, rel_unc_Tobs):
    lp = lnprior(p)
    if not np.isfinite(lp):
        # Return
        return -np.inf
    # Return
    return lp + residual(p, robs, Tobs, rel_unc_Tobs)


r_obs, Tobs = mock_population_old(10000)


ndim     = 2
nwalkers = 50
# first guess
p0 = [[0.9, 0.9] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, 
                                args=(r_obs, Tobs, 0.1))
pos, prob, state  = sampler.run_mcmc(p0, 300, progress=True)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 5000, progress=True)
like    = sampler.flatlnprobability # likelihood
samples = sampler.flatchain # posterior
maxlike = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
print ("ML estimator : " , maxlike)

# Save likelihood
file_object = open("../results/likelihood_old", "wb")
pickle.dump(like, file_object, protocol=2)
file_object.close()

# Save posterior
file_object = open("../results/posterior_old", "wb")
pickle.dump(samples, file_object, protocol=2)
file_object.close()
