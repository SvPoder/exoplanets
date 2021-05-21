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
import matplotlib.pyplot as plt

def lnL_sb(gamma, f, rs, Tobs, robs, Mobs, heat_int, relT, rho0=0.42, v=None):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    T = temperature_withDM(robs, heat_int, f=f, M=Mobs*M_sun.value, 
                           parameters=[gamma, rs, rho0], v=v)
    # return
    return -0.5*np.sum(((T-Tobs)/(relT*Tobs))**2.) 

def lnL_b(Tobs, heat_int, relT):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    # Calculate predicted intrinsic temperature
    T = temperature(heat_int, R_jup.value).value
    # return
    return -0.5*np.sum(((T-Tobs)/(relT*Tobs))**2.) 


def TS(gamma, f, rs, Tobs, robs, Mobs, heat_int, relT, rho0=0.42, v=None):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb(gamma, f, rs, Tobs, robs, Mobs, heat_int, relT, rho0, v)
            -2*lnL_b(Tobs, heat_int, relT))


def p_value_sb(gamma_k, f, rs, Tobs_R, robs_R, Mobs_R, heat_int_R, relT_R, 
               relM_R, relR_R, relA_R, steps=300):
    """
    Return p-value and TS pdf for gamma_k @ (f, rs) under s+b hypothesis
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
        TS_k[i] = TS(gamma_k, f, rs, Tobs, robs, Mobs, heat_int, relT_R)
    # TS pdf @ gamma_k
    counts, bins_ed, _ = plt.hist(TS_k, bins=50, density=True)
    # observed TS
    q_gamma_k_obs = TS(gamma_k, f, rs, Tobs_R, robs_R, Mobs_R, heat_int_R, relT_R)
    # Compute p-value
    pos = np.where(bins_ed > q_gamma_k_obs)
    _p = 0
    for i in range(len(pos[0])):
        _p += counts[-1-i]*(bins_ed[1] + bins_ed[0])
    # return
    return _p#, counts, bins_ed

def p_value_b(gamma_k, f, rs, Tobs_R, robs_R, Mobs_R, heat_int_R, relT_R, 
              relM_R, relR_R, relA_R, steps=300):
    """
    Return p-value and TS pdf for gamma_k @ (f, rs) under b hypothesis
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
        TS_k[i] = TS(gamma_k, f, rs, Tobs, robs, Mobs, heat_int, relT_R)
    # TS pdf @ gamma_k
    #fig, ax = plt.subplots()
    counts, bins_ed, _ = plt.hist(TS_k, bins=50, density=True)
    # observed TS
    q_gamma_k_obs = TS(gamma_k, f, rs, Tobs_R, robs_R, Mobs_R, heat_int_R, relT_R)
    #ax.axvline(q_gamma_k_obs, ls="--", color="b", lw=2.5)
    # Compute p-value
    pos = np.where(bins_ed > q_gamma_k_obs)
    _p = 0
    for i in range(len(pos[0])):
        try:
            _p += counts[-1-i]*(bins_ed[1] + bins_ed[0])
        except:
            _p += 0.
    # return
    return _p#, counts, bins_ed

def UL(f, nBDs, relT_R, relM_R, relR_R, relA_R, steps=300):
    # Generate "real" observation assuming only background (no DM)
    rho0=0.42
    # Load ATMO2020 model
    path   = "/home/mariacst/exoplanets/exoplanets/data/"
    data   = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points = np.transpose(data[0:2, :])
    values = data[2]
    #np.random.seed(42) # ========= reproducable results!
    robs_R, Tobs_R, mass_R, ages_R = mock_population_all(nBDs, relT_R, relM_R, 
                                         relR_R, relA_R,
                                         0, 1., 1., rho0_true=rho0, v=None)
    xi         = np.transpose(np.asarray([ages_R, mass_R]))
    Teff       = griddata(points, values, xi)
    heat_int_R = heat(Teff, np.ones(len(Teff))*R_jup.value)

    rs       = np.asarray([5.])
    gamma_up = np.ones(len(rs))*10

    for i in range(len(rs)):
        if rs[i] > 7.:
            gamma_k = np.linspace(0.5, 3., 30) # change this?
        else:
            gamma_k  = np.linspace(0, 1.5, 25) # change this?
        for g in gamma_k:
            _p_sb = p_value_sb(g, f, rs[i], Tobs_R, robs_R, mass_R, 
                               heat_int_R, relT_R, relM_R, relR_R, relA_R, 
                               steps=steps)
            _p_b = p_value_b(g, f, rs[i], Tobs_R, robs_R, mass_R, heat_int_R, 
                             relT_R, relM_R, relR_R, relA_R, steps=steps)
            try:
                CL = _p_sb / _p_b
            except ZeroDivisionError:
                CL = 200.
            if CL < 0.05:
                gamma_up[i] = g
                #CL_up[i]     = CL
                #print(rs[i], g, CL)
                break
    #return
    return gamma_up

def UL_at_rs(rs, f, nBDs, 
             relT_R, relM_R, relR_R, relA_R, steps=300):
    # Generate "real" observation assuming only background (no DM)
    rho0=0.42
    # Load ATMO2020 model
    path   = "/home/mariacst/exoplanets/exoplanets/data/"
    data   = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)
    points = np.transpose(data[0:2, :])
    values = data[2]
    #np.random.seed(42) # ========= reproducable results!
    robs_R, Tobs_R, mass_R, ages_R = mock_population_all(nBDs, relT_R, relM_R, 
                                         relR_R, relA_R,
                                         0, 1., 1., rho0_true=rho0, v=None)
    xi         = np.transpose(np.asarray([ages_R, mass_R]))
    Teff       = griddata(points, values, xi)
    heat_int_R = heat(Teff, np.ones(len(Teff))*R_jup.value)

    if rs > 7.:
        gamma_k = np.linspace(0.5, 3., 30) # change this?
    else:
        gamma_k  = np.linspace(0, 1.5, 30) # change this?
    for g in gamma_k:
        _p_sb = p_value_sb(g, f, rs, Tobs_R, robs_R, mass_R, 
                           heat_int_R, relT_R, relM_R, relR_R, relA_R, 
                           steps=steps)
        _p_b = p_value_b(g, f, rs, Tobs_R, robs_R, mass_R, heat_int_R, 
                         relT_R, relM_R, relR_R, relA_R, steps=steps)
        try:
            CL = _p_sb / _p_b
        except ZeroDivisionError:
            CL = 200.
        if CL < 0.05:
            gamma_up = g
            break
    #return
    return gamma_up


if __name__=="__main__":
    nBDs=100; relT_R=0.10; relM_R=0.10; relR_R=0.10; relA_R=0.10
    f        = 1.
    steps    = int(sys.argv[1])
    rank     = int(sys.argv[2]) + 100
    rs       = float(sys.argv[3])
    ite      = 12
    gamma_up = np.ones(ite)*100
    for i in range(ite):
        gamma_up[i] = UL_at_rs(rs, f, nBDs, relT_R, relM_R, relR_R, relA_R, steps=steps)
    # save results
    np.savetxt("UL_f%.1f_rs%.1f_%i_steps%i.dat" %(f, rs, rank, steps), 
               gamma_up.T, fmt="%.4f")

