# ===========================================================================
#
# This file contains functions for simulating a mock population of brown
# dwarfs (BDs)
#
# ===========================================================================
import numpy as np
from scipy.interpolate import interp1d, griddata
from astropy.constants import L_sun, R_jup, M_jup, M_sun
from utils import heat, temperature_withDM
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
    ymin = 0.1; ymax = 1.0#R0
    #print("maximimum observed GC distance = ", ymax)
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

def IMF_sampling(alpha, size, Mmin=14, Mmax=55):
    """
    Sampling from power-law distribution
    """
    y = np.random.uniform(0, 1, size=size)
    return ((Mmax**(alpha+1) - Mmin**(alpha+1))*y + Mmin**(alpha+1))**(1./(alpha+1))

def mock_population(N, rel_unc_Tobs, rel_mass, f_true, gamma_true,
                    rs_true, rho0_true=0.42):
    """
    Generate N observed exoplanets

    Assumptions
    -----------
    1) N observed exoplanets distributed according to E2 bulge + BR disc
    2) (All) exoplanets radius = Rjup
    3) BD evolution model taken from ATMO 2020
    4) BDs have masses chosen between 14-55 Mjup assuming power-law IMF and
       unifrom age distribution between 1-10 Gyr
    5) Tobs has relative uncertainty rel_unc_Tobs
    6) Estimated masses have an uncertainty of rel_mass
    """
    #np.random.seed(42)
    _N = int(2*N)
    # galactocentric radius of simulated exoplanets
    r_obs = spatial_sampling(N)
    # Age
    ages = np.random.uniform(1., 10., N) # [yr] / [1-10 Gyr]
    # Mass
    mass = IMF_sampling(-0.6, _N, Mmin=6, Mmax=75) # [Mjup]
    mass = mass*M_jup.value/M_sun.value # [Msun]
    # add Gaussian noise
    mass_wn = mass + np.random.normal(loc=0, scale=(rel_mass*mass), size=_N)
    # select only those objects with masses between 14 and 55 Mjup
    pos  = np.where((mass_wn > 0.013) & (mass_wn < 0.053))

    mass     = mass[pos][:N]
    mass_wn  = mass_wn[pos][:N]
    
    # load theoretical BD cooling model - ATMO 2020
    path =  "./data/"
    #path = "/Users/mariabenito/Dropbox/exoplanets/DM/python/cluster/data/"
    #path  = path 
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
    M = np.sort(M)[5:] # further remove larger masses
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
    #print(len(Teff), len(r_obs), len(heat_int), len(mass))
    # Observed velocity (internal heating + DM)
    Tobs = temperature_withDM(r_obs, heat_int, f=f_true, R=R_jup.value,
                           M=mass*M_sun.value,
                           parameters=[gamma_true, rs_true, rho0_true])
    # add Gaussian noise
    Tobs = Tobs + np.random.normal(loc=0, scale=(rel_unc_Tobs*Tobs), size=N)
    

    #m_obs = np.zeros(len(mass))
    #for i in range(len(mass)):
    #    m_obs[i] = mass[i] + np.random.normal(loc=0, scale=(0.2*mass[i]))
    #    if m_obs[i] > 0.053 or m_obs[i] < 0.013:
    #        while m_obs[i] > 0.053 or m_obs[i] < 0.013:
    #            m_obs[i] = mass[i] + np.random.normal(loc=0, scale=(0.2*mass[i]))

    #return
    return r_obs, Tobs, mass_wn, ages


def mock_population_sens(N, rel_unc_Tobs, rel_mass, 
                         points, values,
                         f_true, gamma_true,
                         rs_true, rho0_true=0.42):
    """
    Generate N observed exoplanets - intended to be run with sensitivity
    analysis

    Assumptions
    -----------
    1) N observed exoplanets distributed according to E2 bulge + BR disc
    2) (All) exoplanets radius = Rjup
    3) BD evolution model taken from ATMO 2020
    4) BDs have masses chosen between 14-55 Mjup assuming power-law IMF and
       unifrom age distribution between 1-10 Gyr
    5) Tobs has relative uncertainty rel_unc_Tobs
    6) Estimated masses have an uncertainty of rel_mass
    """
    #np.random.seed(42)
    _N = int(2*N)
    # galactocentric radius of simulated exoplanets
    r_obs = spatial_sampling(N)
    # Ages and masses of simulated BDs
    ages = np.random.uniform(1., 10., N) # [yr] / [1-10 Gyr]
    mass = IMF_sampling(-0.6, _N, Mmin=6, Mmax=75) # [Mjup]
    mass = mass*M_jup.value/M_sun.value # [Msun]

    # add Gaussian noise
    mass_wn = mass + np.random.normal(loc=0, scale=(rel_mass*mass), size=_N)
    # select only those objects with masses between 14 and 55 Mjup
    pos  = np.where((mass_wn > 0.013) & (mass_wn < 0.053))

    mass     = mass[pos][:N]
    mass_wn  = mass_wn[pos][:N]

    xi = np.transpose(np.asarray([ages, mass]))

    Teff     = griddata(points, values, xi) # true Teff [K]
    heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)
    
    # Observed velocity (internal heating + DM)
    Tobs = temperature_withDM(r_obs, heat_int, f=f_true, R=R_jup.value,
                           M=mass*M_sun.value,
                           parameters=[gamma_true, rs_true, rho0_true])
    # add Gaussian noise
    Tobs = Tobs + np.random.normal(loc=0, scale=(rel_unc_Tobs*Tobs), size=N)
    
    # estimated Teff [K]
    xi = np.transpose(np.asarray([ages, mass_wn]))
    Teff = griddata(points, values, xi)

    #return
    return Tobs, Teff
