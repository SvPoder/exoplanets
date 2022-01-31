import sys
sys.path.append("/home/mariacst/exoplanets/running/.env/lib/python3.6/site-packages")
import emcee
import numpy as np
from scipy.interpolate import griddata
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
from astropy.constants import R_jup
import glob
import pickle
from scipy.interpolate import interp1d
from emcee_functions import lnprob

#start = time.time()

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
path = "/home/mariacst/exoplanets/running/data/"
data = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
points = np.transpose(data[0:2, :])
values = data[2]
# Mock observation
np.random.seed(rank)
(robs, robs_R, sigmarobs, Tobs, Tobs_R, sigmaTobs, Mobs, Mobs_R,
     sigmaMobs, Aobs, Aobs_R, sigmaAobs) = mock_population_all(nBDs, relTobs, sigma,
                                      sigma, sigma, f_true, gamma_true,
                                      rs_true, rho0_true=rho0, Tmin=Tcut, v=v)

#from lmfit import minimize, Parameters

#mass, counts = np.genfromtxt("/home/mariacst/exoplanets/exoplanets/jupyter-notebook/IMF_Tcut650_afterCut_nounc.dat", unpack=True)
#from scipy.interpolate import interp1d
#prior_M = interp1d(mass, counts)

#def objective_function(params, Mobs, sigma_Mobs): 
#    M = params["M_ML"]
#    #print(M.value, -np.exp(-(M-Mobs)**2/(2*sigma_Mobs**2))*prior_M(M))
#    return -np.exp(-(M-Mobs)**2/(2*sigma_Mobs**2))*prior_M(M) + 1

#from astropy.constants import M_sun, M_jup
#M_ML = []
#for i in range(len(Mobs)):
#    params = Parameters()
#    params.add("M_ML", value=55, min=15.4, max=73.8)
#    out = minimize(objective_function, params, args=(Mobs[i]*M_sun.value/M_jup.value, 
#                sigmaMobs[i]*M_sun.value/M_jup.value))  
#    M_ML.append(out.params["M_ML"].value)
#M_ML = np.asarray(M_ML)*M_jup.value/M_sun.value

## calculate predictic intrinsic temperature
xi       = np.transpose(np.asarray([Aobs, Mobs]))
Teff     = griddata(points, values, xi)
# ------------------ RECONSTRUCTION --------------------------------------
# Load variables analytical derivatives Tint
masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)
ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)
a_interp = interp1d(masses, a)
b_interp = interp1d(masses, b)
c_interp = interp1d(ages, c)



ndim     = 3
nwalkers = 150
# first guess
p0 = [[0.9, 0.9, 20.] + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
              args=(robs_R, sigmarobs, Mobs_R, sigmaMobs, Aobs_R, sigmaAobs, Tobs,
                    sigmaTobs, Teff, points, values, a_interp, b_interp, 
                    c_interp, v))

pos, prob, state  = sampler.run_mcmc(p0, 200, progress=True)
sampler.reset()
pos, prob, state  = sampler.run_mcmc(pos, 6000, progress=True)

# Save likelihood
#_path = "/hdfs/local/mariacst/exoplanets/results/likelihood/velocity/v100/analytic/fixedT10Tcut650_factorM/"
#filepath    = (_path + "N%isigma%.1f/like_" %(nBDs, sigma) + ex)
filepath = "./like_" + ex
file_object = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))            
                    + "v" + str(rank), "wb") 
pickle.dump(sampler.flatlnprobability, file_object, protocol=2)
file_object.close() 
# Save posterior
#_path = "/hdfs/local/mariacst/exoplanets/results/posterior/velocity/v100/analytic/fixedT10Tcut650_factorM/"
#filepath    = (_path + "N%isigma%.1f/posterior_" %(nBDs, sigma) + ex)       
filepath = "./posterior_" + ex
file_object2 = open(filepath + ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"
                    %(nBDs, sigma, f_true, gamma_true, rs_true))            
                    + "v" + str(rank), "wb")                                  
pickle.dump(sampler.flatchain, file_object2, protocol=2)
