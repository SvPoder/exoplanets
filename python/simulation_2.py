# =========================================================================== 
#
# This file contains the second toy-model -although more realistic-
# simulation
#
# Assumptions
# -----------
#   1) N observed exoplanets distributed w/ log uniform probability within 
#      0.1 and 8.178
#   2) (All) exoplanets radius = Rjup
#   3) Brown dwarf (BD) evolution model taken from fig 2 of Saumon & Marley'08
#   4) BDs have log(ages) = [9, 9.9] yr & mass = [50 Mjup, 75Mjup] w/ uniform
#      probability
#   5) Relative uncertainty in observed Teff is 10%
#
# =========================================================================== 
import numpy as np
from scipy.stats import loguniform
from scipy.interpolate import interp1d, interp2d
from astropy.constants import L_sun, R_jup, M_jup, M_sun
from utils import temperature, heat, temperature_withDM
# Test
import matplotlib.pyplot as plt

# generation of mock catalog
np.random.seed(42)
# Number of simulated exoplanets
N = 100
# galactocentric radius of simulated exoplanets
r_obs = loguniform.rvs(0.1, 8.178, size=N)
# load theoretical BD cooling model taken from Saumon & Marley '08 (fig 2)
age  = {}
logL = {}
L    = {}
M    = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
Teff = {}

# TODO simplify by directly interpolating on heating/luminosity
for mass in M:
    data = np.genfromtxt("../data/saumon_marley_fig2_" + str(mass) + ".dat", 
                         unpack=True)
    age[mass]  = data[0]
    heat_int   = np.power(10, data[1])*L_sun.value
    Teff[mass] = temperature(heat_int, R_jup)
log_age  = np.linspace(6.1, 9.9, 10)
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
del age, logL, L, M, Teff, _log_age, _mass, _teff
# Ages and masses of simulated BDs
log_ages = np.random.uniform(9., 9.9, N) # [yr] / [1-10 Gyr]
mass     = np.random.uniform(50, 75, N)
mass     = mass*M_jup/M_sun # [Msun]
heat_int = np.zeros(N)
for i in range(N):
    heat_int[i] = heat(Teff_interp_2d(log_ages[i], mass[i]), R_jup.value)

# TODO check t_obs calculation is correct
t_obs = temperature_withDM(r_obs, heat_int, f=1, R=R_jup.value, 
                           M=mass*M_sun.value, parameters=[1, 20, 0.42])
t_obs = t_obs + 0.1*np.random.normal(loc=0, scale=(0.1*t_obs), size=N)


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(r_obs, t_obs, color="k", s=5)
#ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel("Temperature [K]")
ax.set_xlabel("Galactocentric distance [kpc]")
plt.show()
