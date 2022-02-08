import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
sys.path.append("/home/sven/exoplanetenv/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata, interp1d
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
from astropy.constants import R_jup, M_jup, G, sigma_sb
import glob
import pickle
import time
from derivatives import derivativeTDM_wrt_M_emcee, derivativeTDM_wrt_r_emcee, derivativeTintana_wrt_A
from utils import T_DM_optimised, temperature_withDM_optimised, gNFW_rho
import os
import matplotlib.pyplot as plt
import corner
from scipy.stats import gaussian_kde, binned_statistic
from scipy.ndimage import gaussian_filter1d

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
Tcut       = 650.
# ------------------------------------------------------------------------
# Load theoretical cooling model
path = "/home/sven/repos/exoplanets-1/data/"
data = np.genfromtxt(path+"ATMO_CEQ_vega_MIRI.txt", unpack=True)
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

print("Input section done.")

# ---------------------- emcee section ----------------------------------------

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

    #dervT_M = 0

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

    _TDM = T_DM_optimised(robs, R=R_jup.value, M=Mobs, Rsun=Rsun, f=f,
         params=[gamma, rs, rho0], v=v, epsilon=epsilon, gNFW_rho = _gNFW_rho)

    # model temperature [K]
    Tmodel = temperature_withDM_optimised(Teff, TDM = _TDM)

    # _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigmarobs, sigmaMobs,
    #                                sigmaAobs, Teff, points, values,
    #                                f, [gamma, rs, rho0], a_interp, b_interp, c_interp, v=v, R=R_jup.value, Rsun=Rsun,
    #                                epsilon=epsilon, TDM = _TDM, gNFW_rho = _gNFW_rho)

    _sigma_Tmodel2 = 0

    # return
    return (-0.5*np.sum(np.log(sigmaTobs**2 + _sigma_Tmodel2) +
                        (Tmodel-Tobs)**2/(sigmaTobs**2 + _sigma_Tmodel2)))

def lnprob(p):
    lp = lnprior(p)

    if not np.isfinite(lp):
        return -np.inf

    return lp + residual(p)

from multiprocessing import Pool
ncpu = 6
ndim     = 3
nwalkers = 10

# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]

debug_results_path = "/home/sven/repos/exoplanets-1/debug_results/" + ex + "/"
os.makedirs(debug_results_path, exist_ok=True)

# Emcee backend support
# TO USE: Uncomment and add 'backend=backend' to sampler instantiation below.
backend_file = debug_results_path + "walkers" +("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true)) + "v" + str(rank) + ".h5"

backend = emcee.backends.HDFBackend(backend_file)
backend.reset(nwalkers, ndim)


with Pool(ncpu) as pool:

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool = pool, backend=backend)

    print("Starting emcee")
    start = time.time()

    steps = 6000

    pos, prob, state  = sampler.run_mcmc(p0, 200, progress=False)
    sampler.reset()

    pos, prob, state  = sampler.run_mcmc(p0, steps, progress=True)

    elapsed = time.time() - start
    print("Finished v{0} gamma: {1} rs: {2} Emcee took ".format(rank, gamma_true, rs_true) + str(elapsed))

# -------------------------------------------------------------------------

# Save likelihood
_path =  "/home/sven/repos/exoplanets-1/like_post_data/" + ex + "/"

os.makedirs(_path, exist_ok=True)

like_path    = (_path + "like_" + ex + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))
                    + "v" + str(rank))

file_object = open(like_path, "wb")
pickle.dump(sampler.flatlnprobability, file_object, protocol=2)
file_object.close()

# Save posterior
post_path   = (_path + "posterior_" + ex + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))
                    + "v" + str(rank))

file_object2 = open(post_path, "wb")
pickle.dump(sampler.flatchain, file_object2, protocol=2)
file_object2.close()

# -----------------------------------------------------------------------

def plot_walkers(samples_data, param_labels, save_fig, plot_path, label_kwrgs=""):

    # Setup parameter labels
    num_parameters = len(param_labels)

    fig, axes = plt.subplots(num_parameters, figsize=(10, 7), sharex=True)

    fig.suptitle('Positions of walkers: {}'.format(label_kwrgs), fontsize=18, y= 0.95)

    for i in range(num_parameters):

        ax = axes[i]

        # Steps, walkers, parameter
        ax.plot(samples_data[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples_data))
        ax.set_ylabel(param_labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("Step number")

    plot_name = "walkers" +("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true)) + "v" + str(rank)

    if(save_fig):
        plt.savefig(plot_path + plot_name + ".png", dpi=150)


def display_emcee_results(file, save_fig, plot_path, label_kwrgs=""):

    param_labels = ['f', 'gamma', 'rs']
    reader = emcee.backends.HDFBackend(file, read_only=True)
    samples_data = reader.get_chain()

    print("Samples shape (steps, walkers, params): {}".format(samples_data.shape))
    plot_walkers(samples_data, param_labels, True, plot_path, label_kwrgs)

    flat_samples = reader.get_chain(discard=200, flat=True)
    print("Flattened samples shape (steps x walkers, params): {}".format(flat_samples.shape))

    corner.corner(flat_samples, labels=param_labels)

    plot_name = "corner" +("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true)) + "v" + str(rank)

    if(save_fig):
        plt.savefig(plot_path + plot_name + ".png", dpi=150)


# Load like and post files
like1     = pickle.load(open(like_path, "rb"))
samples1  = pickle.load(open(post_path, "rb"))

# Plot Likelihood
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

axes.set_xlabel(r"$\gamma$")
axes.set_ylabel(r"-ln($\mathcal{L}$)")

bin_n=40
x = binned_statistic(samples1[:, 1], like1, 'max', bins=bin_n)[1]
y = binned_statistic(samples1[:, 1], like1, 'max', bins=bin_n+1)[0]
y = y - np.max(y[~np.isnan(y)]) + 1
axes.plot(x, y, ls="-", color="orange", lw=2.5)
axes.axvline(gamma_true, color="orange", ls="--", label="True gamma")

axes.set_ylim(-1, 1.5)
axes.legend(fontsize=16)

plot_title = "Likelihood" +("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"%(nBDs, sigma, f_true, gamma_true, rs_true)) + "v" + str(rank)
plt.title(plot_title)
plt.savefig(debug_results_path + plot_title + ".png", dpi=150)


# Plot Posterior
fig, axes = plt.subplots(1, 1, figsize=(5, 5))

axes.set_xlabel(r"$\gamma$")
axes.set_ylabel(r"Posterior")

bin_n=80
axes.axvline(gamma_true, color="orange", ls="--", label="True gamma")

kde      = gaussian_kde(samples1[:, 1])
xvals = binned_statistic(samples1[:, 1], like1, 'mean', bins=bin_n)[1]
axes.plot(xvals, kde(xvals)/np.max(kde(xvals)), color="red", lw=2.)

plot_title = "Posterior" +("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"%(nBDs, sigma, f_true, gamma_true, rs_true)) + "v" + str(rank)
plt.title(plot_title)
plt.savefig(debug_results_path + plot_title + ".png", dpi=150)

# Plot Walkers and Corner
display_emcee_results(backend_file, True, debug_results_path)



