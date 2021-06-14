import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
sys.path.append("/home/mariacst/exoplanets/exoplanets/python/")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all
import numpy as np
from scipy.interpolate import griddata
from utils import heat, temperature_withDM, temperature
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore


def lnL_sb(gamma, f, rs, Tobs, robs, Mobs, heat_int, rho0=0.42, v=None):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    T = temperature_withDM(robs, heat_int, f=f, M=Mobs*M_sun.value, 
                           parameters=[gamma, rs, rho0], v=v)
    # return
    return -0.5*np.sum(((T-Tobs)/(0.1*Tobs))**2.) 

def lnL_b(Tobs, heat_int):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    # Calculate predicted intrinsic temperature
    T = temperature(heat_int, R_jup.value).value
    # return
    return -0.5*np.sum(((T-Tobs)/(0.1*Tobs))**2.) 

def TS(gamma, f, rs, Tobs, robs, Mobs, heat_int, rho0=0.42, v=None):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb(gamma, f, rs, Tobs, robs, Mobs, heat_int, rho0, v)
            -2*lnL_b(Tobs, heat_int))

def p_value_sb(gamma_k, f, rs, nBDs, Tobs_R, robs_R, Mobs_R, heat_int_R, relT_R,
               relM_R, relR_R, relA_R, steps=300):
    """
    Return p-value for gamma_k @ (f, rs) under s+b hypothesis
    """
    # Compute TS pdf
    TS_k  = np.zeros(steps)
    # Load ATMO2020 model
    path   = "/home/mariacst/exoplanets/exoplanets/data/"
    data   = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points = np.transpose(data[0:2, :])
    values = data[2]
    
    for i in range(steps):
        robs, Tobs, Mobs, ages = mock_population_all(nBDs, relT_R, relM_R, relR_R,
                                                     relA_R, f, gamma_k, rs)
        # Predicted intrinsic temperatures
        xi       = np.transpose(np.asarray([ages, Mobs]))
        Teff     = griddata(points, values, xi)
        heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)
        # TS
        TS_k[i]  = TS(gamma_k, f, rs, Tobs, robs, Mobs, heat_int)
    # observed TS
    q_gamma_k_obs = TS(gamma_k, f, rs, Tobs_R, robs_R, Mobs_R, heat_int_R)
    # return
    return (100-percentileofscore(TS_k, q_gamma_k_obs, kind="strict"))

def p_value_b(gamma_k, f, rs, nBDs, Tobs_R, robs_R, Mobs_R, heat_int_R, relT_R,
              relM_R, relR_R, relA_R, steps=300):
    """                                                                        
    Return p-value for gamma_k @ (f, rs) under b hypothesis
    """                                                                        
    # Compute TS pdf                                                           
    TS_k  = np.zeros(steps)                                                    
    # Load ATMO2020 model                                                      
    path   = "/home/mariacst/exoplanets/exoplanets/data/"                      
    data   = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)     
    points = np.transpose(data[0:2, :])                                        
    values = data[2]                                                           
                                                                               
    for i in range(steps):                                                     
        # Generate experiments under s+b hypothesis
        robs, Tobs, Mobs, ages = mock_population_all(nBDs, relT_R, relM_R, relR_R,
                                         relA_R, 0., gamma_k, rs)              
        # Predicted intrinsic temperatures
        xi       = np.transpose(np.asarray([ages, Mobs]))
        Teff     = griddata(points, values, xi)
        heat_int = heat(Teff, np.ones(len(Teff))*R_jup.value)
        # TS
        TS_k[i] = TS(gamma_k, f, rs, Tobs, robs, Mobs, heat_int)
    # observed TS
    q_gamma_k_obs = TS(gamma_k, f, rs, Tobs_R, robs_R, Mobs_R, heat_int_R)
    # return
    return (100-percentileofscore(TS_k, q_gamma_k_obs, kind="strict"))

def UL(rs, f, nBDs, relT_R, relM_R, relR_R, relA_R, steps=300):
    # Generate "real" observation assuming only background (no DM)
    rho0=0.42
    # Load ATMO2020 model
    path   = "/home/mariacst/exoplanets/exoplanets/data/"
    data   = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points = np.transpose(data[0:2, :])
    values = data[2]
    robs_R, Tobs_R, mass_R, ages_R = mock_population_all(nBDs, 0., 0., 
                                         0., 0.,
                                         0, 1., 1., rho0_true=rho0, v=None)
    xi         = np.transpose(np.asarray([ages_R, mass_R]))
    Teff       = griddata(points, values, xi)
    heat_int_R = heat(Teff, np.ones(len(Teff))*R_jup.value)

    import pdb
    pdb.set_trace()

    gamma_up = np.ones(len(rs))*10
    for i in range(len(rs)):
        if rs[i] > 7.:
            gamma_k = np.linspace(0.4, 2.9, 35) # change this?
        else:
            gamma_k  = np.linspace(0, 1.5, 35) # change this?
        for g in gamma_k:
            _p_sb = p_value_sb(g, f, rs[i], nBDs, Tobs_R, robs_R, mass_R, 
                               heat_int_R, relT_R, relM_R, relR_R, relA_R, 
                               steps=steps)
            _p_b = p_value_b(g, f, rs[i], nBDs, Tobs_R, robs_R, mass_R, heat_int_R,
                             relT_R, relM_R, relR_R, relA_R, steps=steps)
            try:
                CL = _p_sb / _p_b
            except ZeroDivisionError:
                CL = 200.
            if CL < 0.05:
                print(rs[i], g)
                gamma_up[i] = g
                break
    #return
    return gamma_up


if __name__=="__main__":
    np.random.seed(42) # ====== reproducable results
    nBDs=100; relT_R=0.1; relM_R=0.1; relR_R=0.1; relA_R=0.1
    f        = float(sys.argv[1])
    steps    = int(sys.argv[2])
    rs       = np.asarray([5., 10., 15., 20.])
    gamma_up = UL(rs, f, nBDs, relT_R, relM_R, relR_R, relA_R, steps=steps)
    # save results
    np.savetxt("UL_f%.1f_Asimov42_steps%i.dat" %(f, steps), gamma_up, fmt="%.4f")

