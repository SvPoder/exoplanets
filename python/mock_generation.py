# ===========================================================================
#
# This file contains functions for simulating a mock population of brown
# dwarfs (BDs)
#
# ===========================================================================
import numpy as np
from scipy.interpolate import interp1d, griddata
from astropy.constants import L_sun, R_jup, M_jup, M_sun
from utils import heat, temperature_withDM, random_powerlaw
import glob

def rho_bulge(r, phi, theta, R0=8.178, x0=0.899, y0=0.386, z0=0.250, 
              alpha=0.415):
    """
    Density profile for Stanek + '97 (E2) bulge [arbitrary units]
    (all spatial coordiantes are given in kpc)
    """
    x0 = x0*R0/8. # rescale to adopted R0 value
    y0 = y0*R0/8. 
    # return
    return (np.exp(-np.sqrt(np.sin(theta)**2*((np.cos(phi+alpha)/x0)**2 +
                            (np.sin(phi+alpha)/y0)**2) + 
                            (np.cos(theta)/z0)**2)*r))

def rho_disc(r, theta, R0=8.178, Rd=2.15, zh=0.40):
    """
    Density profile for Bovy and Rix disc [arbitrary units]
    (all spatial coordiantes are given in kpc)
    """
    Rd = Rd*R0/8. # rescale to adopted R0 value
    # return
    return np.exp(-r*np.sin(theta)/Rd)*np.exp(-r*np.cos(theta)/zh)


def rho(r, phi, theta, R0=8.178):
    """
    Density profile [arbitrary units]
    """
    # continuity condition at r = 1 kpc
    C    = rho_disc(1., theta, R0)/rho_bulge(1., phi, theta, R0)
    _rho = C*rho_bulge(r, phi, theta, R0)
    # return
    return (np.heaviside(1.-r, 1.)*_rho + 
            np.heaviside(r-1., 0.)*rho_disc(r, theta, R0))

def spatial_sampling(nBDs, phi=0., theta=np.pi/2., R0=8.178):
    """
    Sampling nBDs points from density profile rho using Von Neumann 
    acceptance-rejection technique
    """
    ymin = 0.1; ymax = R0
    umin = np.min([rho(ymin, phi, theta), rho(1., phi, theta), 
                   rho(R0, phi, theta)])
    umax = np.max([rho(ymin, phi, theta), rho(1., phi, theta), 
                   rho(R0, phi, theta)])
    
    i = 0
    r = np.ones(nBDs)*100
    while i<nBDs:
        yi = np.random.uniform(ymin, ymax)
        ui = np.random.uniform(umin, umax)
        if ui < rho(yi, phi, theta, R0):
            r[i] = yi
            i+=1
        else:
            continue
    # return 
    return r

def mock_population(N, rel_unc_Tobs, rel_mass, f_true, gamma_true,
                    rs_true=20, rho0_true=0.42, points=None, values=None):
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
    5) Tobs has relative uncertainty rel_unc_Tobs
    6) Estimated masses have an uncertainty of rel_mass
    """
    np.random.seed(42)
    # galactocentric radius of simulated exoplanets
    r_obs = spatial_sampling(N)
    
    # load theoretical BD cooling model - ATMO 2020
    if points is None:
        path = "./data/"
        path = "../data/evolution_models/ATMO_2020_models/evolutionary_tracks/ATMO_CEQ/"
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

    # Ages and masses of simulated BDs
    ages = np.random.uniform(1., 10., N) # [yr] / [1-10 Gyr]
    mass = random_powerlaw(-0.6, N, Mmin=14, Mmax=55) # [Mjup]
    mass = mass*M_jup/M_sun # [Msun]
    xi = np.transpose(np.asarray([ages, mass]))

    Teff_interp_2d = griddata(points, values, xi)

    # Add uncertainty to mass
    mass_obs = mass + np.random.normal(loc=0, scale=(rel_mass*mass), size=N)
    # Mapping (mass, age) -> Teff -> internal heat flow (no DM)
    heat_int = np.zeros(N)
    Teff     = np.zeros(N)
    for i in range(N):
        Teff[i]     = Teff_interp_2d[i]
        heat_int[i] = heat(Teff[i], R_jup.value)
    
    # Observed velocity (internal heating + DM)
    Tobs = temperature_withDM(r_obs, heat_int, f=f_true, R=R_jup.value,
                           M=mass*M_sun.value,
                           parameters=[gamma_true, rs_true, rho0_true])
    # add uncertainty to temperature
    Tobs = Tobs + np.random.normal(loc=0, scale=(rel_unc_Tobs*Tobs), size=N)
    #return
    return r_obs, Tobs, rel_unc_Tobs, Teff, mass_obs, ages


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
