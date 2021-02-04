import numpy as np
from scipy.stats import chisquare
from scipy.stats import chi2


def sensitivity(r_obs, Tobs, Teff, Nbins=20, r_min=0.1, r_max=8.178):
    """
    Perform goodness-of-fit test and returns True/False if theretical cooling 
    models can/cannot explain observations
    """
    residuals = Tobs - Teff
    r_bins    = np.logspace(np.log10(r_min), np.log10(r_max), Nbins)
    mean      = np.zeros(len(r_bins)-1)
    #mean_exp  = np.zeros(len(r_bins)-1)
    std_exp   = np.zeros(len(r_bins)-1)

    for i in range(len(r_bins)-1):
        pos        = np.where((r_obs > r_bins[i]) & (r_obs < r_bins[i+1]))
        mean[i]    = np.mean(residuals[pos])
        std_exp[i] = np.std(residuals[pos])
    
    _chi2 = np.sum(np.power(mean/std_exp, 2))
    print(chi2.sf(_chi2, len(mean)))

    return
