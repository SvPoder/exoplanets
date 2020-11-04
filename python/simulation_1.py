# =========================================================================== 
#
# This file contains -not complete yet- the first back-of-the-envelope
# toy simulation
#
# =========================================================================== 
import numpy as np
from scipy.stats import loguniform
from utils import temperature_withDM
from astropy.constants import R_jup, M_jup
import matplotlib.pyplot as plt


# generation of mock catalog
np.random.seed(42)
r_obs       = loguniform.rvs(0.1, 8.178, size=200)
gamma_model = 1
f_model     = 1
M_model     = 75*M_jup.value
R_model     = R_jup.value
heat_int    = 1.1e21 # W
t_fiducial  = temperature_withDM(r_obs, heat_int, f=f_model, R=R_model, 
                          M=M_model, 
                          parameters=[gamma_model, 20., 0.42], epsilon=1)
sigma_obs = 0.1*t_fiducial
t_obs     = t_fiducial + 0.1*np.random.normal(loc=0, scale=(0.1*t_fiducial), 
                                              size=len(t_fiducial))


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(r_obs, t_obs, color="k", s=5)
ax.set_xscale("log")
ax.set_ylabel("Temperature [K]")
ax.set_xlabel("Galactocentric distance [kpc]")
plt.show()



