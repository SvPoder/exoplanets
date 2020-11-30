import numpy as np
from scipy.interpolate import interp1d, interp2d
from utils import temperature, heat, temperature_withDM
from mock_generation import mock_population
from astropy.constants import R_jup, M_jup, M_sun, L_sun
import pickle

# Local DM density
rho0 = 0.42 # GeV/cm3

## mock sample of BDs
r_obs, Tobs, rel_unc_Tobs, Teff, mass, log_ages = mock_population(10000)
sigmaTobs = rel_unc_Tobs*Tobs

## load theoretical BD cooling model taken from Saumon & Marley '08 (fig 2)
age  = {}; logL = {}; L = {}; Teff = {}
M    = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
# TODO simplify by directly interpolating heating/luminosity
filepath = "../data/evolution_models/SM08/"
for m in M:
    data = np.genfromtxt(filepath + "saumon_marley_fig2_" + str(m) + ".dat",
                             unpack=True)
    age[m]   = data[0]
    heat_int = np.power(10, data[1])*L_sun.value
    Teff[m]  = temperature(heat_int, R_jup)
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

## calculate predictic intrinsic heat flow for mock BDs
heat_int = np.zeros(len(r_obs))
for i in range(len(r_obs)):
    heat_int[i] = heat(Teff_interp_2d(log_ages[i], mass[i]), R_jup.value)
    #if i < 10:
    #    print(log_ages[i], mass[i], heat_int[i])

# Grid (f, gamma)
step  = 0.01
f     = np.arange(0, 1+step, step)
step  = 0.01
gamma = np.arange(0.01, 2+step, step)
chi2  = np.zeros((len(gamma), len(f)))


for i in range(len(gamma)):
    for j in range(len(f)):
        Tmodel = temperature_withDM(r_obs, heat_int, f=f[j], M=mass*M_sun.value,
                                parameters=[gamma[i], 20., rho0])

        #print(Tobs - Tmodel, sigmaTobs)
        chi2[i][j] = np.sum(np.power((Tobs - Tmodel)/sigmaTobs, 2))

print("Relative uncertainty in Tobs is ", rel_unc_Tobs)

output = open("../results/frequentist/chi2_game0.dat", "wb")
np.save(output, chi2)
