import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
sys.path.append("/home/sven/exoplanetenv/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata, interp1d
from derivatives import derivativeTint_wrt_A, derivativeTint_wrt_M
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
from astropy.constants import R_jup, M_jup, G, sigma_sb
import glob
import pickle
import time

start = time.time()

# Constant parameters & conversions ==========================================
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
_sigma_sb = sigma_sb.value
_G        = G.value
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
Tcut       = 650.
# ------------------------------------------------------------------------
# Load theoretical cooling model
path = "/home/mariacst/exoplanets/running/data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]

# Mock observation
np.random.seed(rank)
(robs, sigmarobs, Tobs, sigmaTobs, Mobs,
     sigmaMobs, Aobs, sigmaAobs) = mock_population_all(nBDs, relTobs, sigma,
                                      sigma, sigma, f_true, gamma_true,
                                      rs_true, rho0_true=rho0, Tmin=Tcut, v=v)

## calculate predictic intrinsic temperature
xi       = np.transpose(np.asarray([Aobs, Mobs]))
Teff     = griddata(points, values, xi)


# Load variables analytical derivatives Tint
masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)
ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)
a_interp = interp1d(masses, a)
b_interp = interp1d(masses, b)
c_interp = interp1d(ages, c)

elapsed = time.time() - start
print("Input section took " + str(elapsed))

# ---------------------- emcee section ----------------------------------------

from derivatives import derivativeTDM_wrt_M_emcee, derivativeTDM_wrt_r_emcee, derivativeTintana_wrt_A
from utils import T_DM, temperature_withDM, gNFW_rho, vc

def T_DM_optimised(r, R=R_jup.value, M=M_jup.value, Rsun=8.178, f=1.,
         params=[1., 20., 0.42], v=None, epsilon=1., gNFW_rho = 0):
    """
    DM temperature
    """
    # escape velocity
    vesc   = np.sqrt(2*_G*M/R)*1e-3 # km/s
    if v:
        _vD = v
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, params) # km/s

    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho # GeV/cm3
    # return
    return np.power((f*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                    conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.)

def temperature_withDM_optimised(r, Tint, R=R_jup.value, M=M_jup.value,
                       f=1., p=[1., 20., 0.42], v=None, Rsun=8.178, epsilon=1, TDM =0):
    """
    Exoplanet temperature : internal heating + DM heating
    """
    return (np.power(np.power(Tint, 4) +
                     np.power(TDM, 4)
                     , 0.25))

# Constant parameters & conversions ==========================================
conversion_into_K_vs_kg = 1.60217e-7
conversion_into_w       = 0.16021766
conv_Msun_to_kg         = 1.98841e+30 # [kg/Msun]
rho0                    = 0.42 # Local DM density [GeV/cm3]
epsilon                 = 1.
Rsun                    = 8.178 # Sun galactocentric distance [kpc]
# ============================================================================

def lnprior(p):
    f, gamma, rs = p
    if ( 0. < gamma < 3. and 0.01 < f < 2. and 0.01 < rs < 70.):
        return 0.
    return -np.inf

def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,
                  Tint, points, values, f, params, a, b, c,
                  v=None, R=R_jup.value, Rsun=8.178, epsilon=1, TDM = 0, gNFW_rho = 0):
    """
    Return squared uncertainty in model temperature [UNITS??]

    Input:
        r : Galactocentric distance [kpc]
        M : mass [Msun]
        A : age [Gyr]

    Assumption: uncertainties in age, mass and galactocentric distance
        are independent
    """

    _TDM = TDM
    Ttot = np.power(_TDM**4 + Tint**4, 0.25)

    TintDivTtot = (Tint/Ttot)**3
    TDMDivTtot = (_TDM/Ttot)**3

    dervT_M = (TintDivTtot* c(A) +
               TDMDivTtot*derivativeTDM_wrt_M_emcee(r, f, params, M, v=v, R=R,
                                                  Rsun=Rsun,epsilon=epsilon, gNFW_rho=gNFW_rho))
    # return
    return (np.power(TintDivTtot*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+
            np.power(dervT_M*sigma_M, 2)+
            np.power(TDMDivTtot*derivativeTDM_wrt_r_emcee(r, f, params, M, v=v,
                                  R=R, Rsun=Rsun, epsilon=epsilon, gNFW_rho=gNFW_rho)*sigma_r, 2))


def residual(p):
    """
    Log likelihood function (without normalization!)
    """
    # unroll free parameters
    f, gamma, rs = p

    _gNFW_rho = gNFW_rho(Rsun, robs, [gamma, rs, rho0])
    #print("Params {} {} {}".format(Rsun, robs, [gamma, rs, rho0]))
    #print("NFW: {}".format(_gNFW_rho))

    _TDM = T_DM_optimised(robs, R=R_jup.value, M=Mobs*conv_Msun_to_kg, Rsun=Rsun, f=f,
         params=[gamma, rs, rho0], v=v, epsilon=epsilon, gNFW_rho = _gNFW_rho)
    #print("TDM: {}".format(_TDM))

    # model temperature [K]
    Tmodel = temperature_withDM_optimised(robs, Teff, M=Mobs*conv_Msun_to_kg, f=f,
                                p=[gamma, rs, rho0], v=v, TDM = _TDM)

    #print("Tmodel: {}".format(Tmodel))
    _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigmarobs, sigmaMobs,
                                   sigmaAobs, Teff, points, values,
                                   f, [gamma, rs, rho0], a_interp, b_interp, c_interp, v=v, R=R_jup.value, Rsun=Rsun,
                                   epsilon=epsilon, TDM = _TDM, gNFW_rho = _gNFW_rho)

    #print("Sigma Tmodel2: {}".format(_sigma_Tmodel2))

    #res = (-0.5*np.sum(np.log(sigmaTobs**2 + _sigma_Tmodel2) +
                        #(Tmodel-Tobs)**2/(sigmaTobs**2 + _sigma_Tmodel2)))
    #print("Residual: {}".format(res))

    # return
    return (-0.5*np.sum(np.log(sigmaTobs**2 + _sigma_Tmodel2) +
                        (Tmodel-Tobs)**2/(sigmaTobs**2 + _sigma_Tmodel2)))


def lnprob(p):
    lp = lnprior(p)

    if not np.isfinite(lp):
        # Return
        return -np.inf

    # Return
    return lp + residual(p)

# ------------------ RECONSTRUCTION --------------------------------------
from multiprocessing import Pool
ncpu = 6
ndim     = 3
nwalkers = 150

# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]

# Backend support
# TO USE: Uncomment and add 'backend=backend' t0 sampler instantiation below.

# backend_file = "walkers_" + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
#                     %(nBDs, sigma, f_true, gamma_true, rs_true)) + "v" + str(rank) + ".h5"
# backend = emcee.backends.HDFBackend(backend_file)
# backend.reset(nwalkers, ndim)

# -------------------------------------------------------------------------

with Pool(ncpu) as pool:

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool = pool)

    print("Starting emcee")
    start = time.time()

    steps = 6000

    pos, prob, state  = sampler.run_mcmc(p0, 200, progress=False)
    sampler.reset()

    pos, prob, state  = sampler.run_mcmc(p0, steps, progress=True)

    elapsed = time.time() - start
    print("Finished v{0} gamma: {1} rs: {2} Emcee took ".format(rank, gamma_true, rs_true) + str(elapsed))


# Save likelihood
_path = "/hdfs/local/sven/exoplanets/walkers/"
_path =  "/hdfs/local/sven/exoplanets/repo_test/"
filepath    = (_path + "like_noburn_" + ex)
file_object = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(sampler.flatlnprobability, file_object, protocol=2)
file_object.close()

# Save posterior
#_path = "/hdfs/local/mariacst/exoplanets/results/posterior/velocity/v100/analytic/fixedT10Tcut650_nocutTwn/"
filepath    = (_path + "posterior_noburn_" + ex)
file_object2 = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))
                    + "v" + str(rank), "wb")
pickle.dump(sampler.flatchain, file_object2, protocol=2)
file_object2.close()

