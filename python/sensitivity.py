import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import brentq
from mock_generation_old import mock_population
from astropy.constants import L_sun, R_jup
from utils import temperature

def interp_find_x(x, y, p_interp):
    try:
        # return
        return p_interp(x) - y
    except:
        return np.inf

def sensitivity(Tobs, Teff, alpha=0.05):
    """
    Perform goodness-of-fit test and returns True/False if theretical cooling 
    models can/cannot explain observations
    """
    # Make number of counts histogram
    ## Calculate bins with equal probability

    p, bins, _ = plt.hist(Teff, bins=40, density=True, cumulative=True)
    p          = np.insert(p, 0, 0)
    p    = np.insert(p, len(p), 1)
    bins = np.insert(bins, len(bins), bins[len(bins)-1]+1000)
    x0   = bins[1]
    p    = np.insert(p, 0, 0)
    bins = np.insert(bins, 0, 0)

    p_interp   = interp1d(bins, p)
    y_h        = np.linspace(0.1, 0.9, 9) # total number of bins = 10
    bins_equal = []
    bins_equal.append(x0)
    for y in y_h:
        root = brentq(interp_find_x, x0, bins[-2], args=(y, p_interp))
        x0 = root
        bins_equal.append(root)
    bins_equal.append(bins[-2])

    n_th, _ = np.histogram(Teff, bins=bins_equal) # theoretical counts
    n, _    = np.histogram(Tobs, bins=bins_equal)

    _chi2   = np.sum(np.power(n-n_th, 2)/n_th) # observed counts
    p_value = chi2.sf(_chi2, len(n)-1)
    #print(p_value)
    if p_value >= alpha:
        # return
        return 1
    else:
        # return
        return 0

def sensitivity_nBDs_relunc(filepath, nBDs, rel_unc, relM, Teff_inter, rank=100):
    
    f     = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamma = [0.2, 0.6, 1, 1.4, 1.8]
    _sens = np.ones((len(f), len(gamma)))*1000
    j = 0
    for _f in f:
        k = 0
        for _g in gamma:
            print("====== nBDS=%i, rel_unc=%.2f, relM=%.2f, f=%.1f, gamma=%.1f" 
                    %(nBDs, rel_unc, relM, _f, _g))
            _bool = np.ones(rank)*100
            for i in range(rank):
                Tobs, Teff = mock_population(nBDs, rel_unc, relM, _f, _g, 
                                             Teff_inter)
                _bool[i] = sensitivity(Tobs, Teff)
            print("Accepted H0 : %i" %int(np.sum(_bool)))
            print("Rejected H0 : %i" %(len(_bool)-int(np.sum(_bool))))
            _sens[j, k] = int(np.sum(_bool))
            k+=1
        j+=1
    # save acceptance ratio out of rank
    np.savetxt(filepath + ("sensitivity_ex3_N%i_relunc%.2f_relM%.2f" 
                           %(nBDs, rel_unc, relM)), _sens)
    # return
    return


if __name__ == '__main__':
    
    nBDs    = [100, 10000]
    rel_unc = [0.05, 0.10]
    relM    = [0.10, 0.20]
    # ------------------------------------------------------------------------
    # load theoretical BD cooling model taken from Saumon & Marley '08 (fig 2)
    age = {}; logL = {}; L = {}; Teff = {}
    M   = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    filepath = "./cluster/data/"
    #filepath = "/Users/mariabenito/Dropbox/exoplanets/DM/python/cluster/data/"
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
    # ------------------------------------------------------------------------
    filepath = "/Users/mariabenito/Desktop/results/ex3/"
    for n in nBDs:
        for rel in rel_unc:
            for rM in relM:
                print(n, rel, rM)
                sensitivity_nBDs_relunc(filepath, n, rel, rM, Teff_interp_2d)
    
