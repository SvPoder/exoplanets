import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton

def interp_find_x(x, y, p_interp):
    # return
    return p_interp(x) - y

def sensitivity(Tobs, Teff, alpha=0.05):
    """
    Perform goodness-of-fit test and returns True/False if theretical cooling 
    models can/cannot explain observations
    """
    # Make number of counts histogram
    ## Calculate bins with equal probability

    p, bins, _ = plt.hist(Teff, bins=40, density=True, cumulative=True)
    p          = np.insert(p, 0, 0)
    p_interp   = interp1d(bins, p)
    y_h        = np.linspace(0, 1, 11) # total number of bins = 10
    bins_equal = []
    for y in y_h:
        bins_equal.append(newton(interp_find_x, 400, args=(y, p_interp)))
    # n_th = 0.10*nBDs (or similar)
    n_th, _ = np.histogram(Teff, bins=bins_equal) # theoretical counts
    #print(n_th)
    n, _    = np.histogram(Tobs, bins=bins_equal)

    _chi2   = np.sum(np.power(n-n_th, 2)/n_th) # observed counts
    p_value = chi2.sf(_chi2, len(n)-1)
    #print(p_value)
    if p_value >= alpha:
        # return
        return True
    else:
        # return
        return False
