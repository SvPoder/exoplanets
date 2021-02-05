import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import newton
from mock_generation import mock_population


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
    p_interp   = interp1d(bins, p)
    y_h        = np.linspace(0, 1, 11) # total number of bins = 10
    bins_equal = []
    for y in y_h:
        bins_equal.append(newton(interp_find_x, 700, args=(y, p_interp)))
    # n_th = 0.10*nBDs (or similar)
    n_th, _ = np.histogram(Teff, bins=bins_equal) # theoretical counts
    #print(n_th)
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


if __name__ == '__main__':

    nBDs = 100
    rel_unc = 0.05
    f     = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamma = [0.2, 0.6, 1, 1.4, 1.8]
    
    for _f in f:
        for _g in gamma:
            print("====== nBDS=%i, rel_unc=%.2f, f=%.1f, gamma=%.1f" 
                    %(nBDs, rel_unc, _f, _g))
            N = 2
            _bool = np.ones(N)*100
            for i in range(N):
                _, Tobs, _, Teff, _, _ = mock_population(nBDs, rel_unc, _f, _g)
                _bool[i] = sensitivity(Tobs, Teff)
            print("Accepted H0 : %i" %int(np.sum(_bool)))
            print("Rejected H0 : %i" %(len(_bool)-int(np.sum(_bool))))
