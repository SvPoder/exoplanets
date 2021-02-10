# ===========================================================================
#
# This file contains functions for simulating a mock population of brown
# dwarfs (BDs)
#
# ===========================================================================
import numpy as np
from scipy.stats import loguniform
from scipy.interpolate import interp1d, interp2d
from astropy.constants import L_sun, R_jup, M_jup, M_sun
from utils import temperature, heat, temperature_withDM, random_powerlaw


def mock_population(N, rel_unc_Tobs, f_true, gamma_true,
                    rs_true=20, rho0_true=0.42):
    """
    Generate N observed exoplanets

    Assumptions
    -----------
    1) N observed exoplanets distributed w/ log uniform probability within
       0.1 and 8.178
    2) (All) exoplanets radius = Rjup
    3) BD evolution model taken from fig 2 of Saumon & Marley'08
    4) BDs have masses chosen between 14-55 Mjup assuming power-law IMF and
       unifrom age distribution between 1-10 Gyr
    5) Tobs has relative uncertainty rel_unc_Tobs - NOT YET!!
    """
    #np.random.seed(42)
    # galactocentric radius of simulated exoplanets
    r_obs = loguniform.rvs(0.1, 8.178, size=N)
    # load theoretical BD cooling model taken from Saumon & Marley '08 (fig 2)
    age = {}; logL = {}; L = {}; Teff = {}
    M   = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    filepath = "./data/"
    # TODO simplify by directly interpolating on heating/luminosity
    for mass in M:
        data = np.genfromtxt(filepath+"saumon_marley_fig2_"+str(mass) + ".dat",
                             unpack=True)
        age[mass]  = data[0]
        heat_int   = np.power(10, data[1])*L_sun.value
        Teff[mass] = temperature(heat_int, R_jup)
    log_age  = np.linspace(6.1, 9.92, 10)
    _log_age = []; _mass = []; _teff = []

    for m in M:
        Teff_interp = interp1d(age[m], Teff[m])
        for lage in log_age:
            _log_age.append(lage)
            _mass.append(m)
            _teff.append(Teff_interp(lage))
    # effective temperature (wo DM heating) vs log(age) and mass exoplanet
    Teff_interp_2d = interp2d(_log_age, _mass, _teff)
    # Ages and masses of simulated BDs
    log_ages = np.random.uniform(9., 9.92, N) # [yr] / [1-10 Gyr]
    mass     = random_powerlaw(-0.6, N, Mmin=14, Mmax=55)
    mass     = mass*M_jup/M_sun # [Msun]
    # Mapping (mass, age) -> Teff -> internal heat flow (no DM)
    heat_int = np.zeros(N)
    Teff     = np.zeros(N)
    for i in range(N):
        Teff[i]     = Teff_interp_2d(log_ages[i], mass[i])
        heat_int[i] = heat(Teff_interp_2d(log_ages[i], mass[i]), R_jup.value)
        #if i < 10:
        #    print(log_ages[i], mass[i], heat_int[i])
    # Observed velocity (internal heating + DM)
    Tobs = temperature_withDM(r_obs, heat_int, f=f_true, R=R_jup.value,
                           M=mass*M_sun.value,
                           parameters=[gamma_true, rs_true, rho0_true])
    # add 10% relative uncertainty
    Tobs = Tobs + np.random.normal(loc=0, scale=(rel_unc_Tobs*Tobs), size=N)
    return r_obs, Tobs, rel_unc_Tobs, Teff, mass, log_ages

#r_obs, Tobs, rel_unc_Tobs, Teff, mass, log_ages = mock_population(10000)

"""
verbose=True
if verbose:
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    _, _, _ = ax[0, 0].hist(Teff, bins=30, histtype="step", linewidth=2.5, color="k")
    _, _, _ = ax[0, 0].hist(Tobs, bins=30, histtype="step", linewidth=2.5, color="r")
    ax[0, 0].set_xlabel("Tobs [K]")
    ax[0, 0].set_ylabel("Number exoplanets")

    ax[0, 1].scatter(r_obs, Tobs, s=5, color="r")
    ax[0, 1].set_ylabel("Tobs [K]")

    colors = bokeh.palettes.viridis(5)
    mass_bins = np.linspace(14, 75, 6)*M_jup/M_sun
    for i in range(len(mass_bins)-1):
        pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]))
        ax[1, 0].scatter(r_obs[pos], Tobs[pos], s=5, color=colors[i],
                     label=("%i - %i Mjup" %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup))))
    ax[1, 0].set_ylabel("Tobs [K]")
    ax[1, 0].set_xlabel("galactocentric distance [kpc]")
    ax[1, 0].legend(frameon=True, fontsize=14)

    age_bins = np.linspace(9, 9.92, 6)
    for i in range(len(age_bins)-1):
        pos = np.where((log_ages > age_bins[i]) & (log_ages < age_bins[i+1]))
        ax[1, 1].scatter(r_obs[pos], Tobs[pos], s=5, color=colors[i],
                     label=("%.1f - %.1f Gyr" %(10**age_bins[i]/10**9, 10**age_bins[i+1]/10**9)))
    ax[1, 1].set_xlabel("galactocentric distance [kpc]")
    ax[1, 1].legend(frameon=True, fontsize=14)
    plt.show()

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    mass_bins = np.linspace(14, 75, 4)*M_jup/M_sun
    age_bins = np.linspace(9, 9.92, 4)

    i = 0
    pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]) &
               (log_ages > age_bins[0]) & (log_ages < age_bins[1]))
    ax[0, 0].scatter(r_obs[pos], Tobs[pos]-Teff[pos], s=5,
                label=("%i - %i Mjup / %.1f - %.1f Gyr"
                       %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup),
                         10**age_bins[i]/10**9, 10**age_bins[i+1]/10**9)))

    pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]) &
               (log_ages > age_bins[2]) & (log_ages < age_bins[3]))
    ax[0, 1].scatter(r_obs[pos], Tobs[pos]-Teff[pos], s=5,
                label=("%i - %i Mjup / %.1f - %.1f Gyr"
                       %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup),
                         10**age_bins[2]/10**9, 10**age_bins[3]/10**9)))


    i = 1
    pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]) &
               (log_ages > age_bins[0]) & (log_ages < age_bins[1]))
    ax[1, 0].scatter(r_obs[pos], Tobs[pos]-Teff[pos], s=5,
                label=("%i - %i Mjup / %.1f - %.1f Gyr"
                       %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup),
                         10**age_bins[i]/10**9, 10**age_bins[i+1]/10**9)))

    pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]) &
               (log_ages > age_bins[2]) & (log_ages < age_bins[3]))
    ax[1, 1].scatter(r_obs[pos], Tobs[pos]-Teff[pos], s=5,
                label=("%i - %i Mjup / %.1f - %.1f Gyr"
                       %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup),
                         10**age_bins[2]/10**9, 10**age_bins[3]/10**9)))

    i = 2
    pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]) &
               (log_ages > age_bins[0]) & (log_ages < age_bins[1]))
    ax[2, 0].scatter(r_obs[pos], Tobs[pos]-Teff[pos], s=5,
                label=("%i - %i Mjup / %.1f - %.1f Gyr"
                       %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup),
                         10**age_bins[i]/10**9, 10**age_bins[i+1]/10**9)))

    pos = np.where((mass > mass_bins[i]) & (mass < mass_bins[i+1]) &
               (log_ages > age_bins[2]) & (log_ages < age_bins[3]))
    ax[2, 1].scatter(r_obs[pos], Tobs[pos]-Teff[pos], s=5,
                label=("%i - %i Mjup / %.1f - %.1f Gyr"
                       %(np.round(mass_bins[i]*M_sun/M_jup), np.round(mass_bins[i+1]*M_sun/M_jup),
                         10**age_bins[2]/10**9, 10**age_bins[3]/10**9)))

    for i in range(3):
        for j in range(2):
            ax[i, j].set_ylim([0, 250])
            ax[i, j].set_xlim([0.1, 8.178])
            ax[i, j].set_xscale("log")
            if j==0:
                ax[i, j].set_ylabel("Tobs - Teff")
            ax[i, j].legend(frameon=True, fontsize=16)
    plt.show()
"""
