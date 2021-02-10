import numpy as np
import pickle

def statistics(filepath, nBDs, rel_unc, f, gamma, rank=100, D=2):
    """
    Calculate mean, median, MAP & ML point estimates

    Inputs
    ------
        filepath    : directory where to save point estimates
        rel_unc_Tobs: relative uncertainty Tobs
        rank        : number of simulations
        D           : dimension parameter space
    """
    mean   = np.zeros((D, rank))
    median = np.zeros((D, rank))
    MAP    = np.zeros((D, rank))
    ML     = np.zeros((D, rank))
    
    for i in range(rank):
        # load posterior + likelihood
        file_name = (filepath + ("posterior_ex1_N%i_relunc%.2f_f%.1fgamma%.1fv%i"
                                 %(nBDs, rel_unc, f, gamma, i)))
        samples   = pickle.load(open(file_name, "rb"))

        file_name = (filepath + ("likelihood_ex1_N%i_relunc%.2f_f%.1fgamma%.1fv%i"
                                 %(nBDs, rel_unc, f, gamma, i)))
        like      = pickle.load(open(file_name, "rb"))
        # calculate point estimates
        for j in range(D):
            mean[j][i]   = np.mean(samples[:, j])
            median[j][i] = np.percentile(samples[:, j], [50], axis=0)
            #TODO need to change # bins to see if results differ
            _n, _bins    = np.histogram(samples[:, j], bins=50)
            MAP[j][i]    = _bins[np.argmax(_n)]
            ML[j][i]     = samples[:, j][np.argmax(like)]

    output = open(filepath + ("statistics_ex1_N%i_relunc%.2f_f%.1fgamma%.1f" 
                              %(nBDs, rel_unc, f, gamma)), 
                  "w")
    for i in range(rank):
        for j in range(D):
            output.write("%.4f  " %mean[j][i])
        for j in range(D):
            output.write("%.4f  " %median[j][i])
        for j in range(D):
            output.write("%.4f  " %MAP[j][i])
        for j in range(D):
            output.write("%.4f  " %ML[j][i])
        output.write("\n")
    output.close()

    # return
    return


if __name__ == '__main__':
    filepath = "../results/bayesian/ex1/N10000_relunc0.10/"
    nBDs     = 10000
    rel_unc  = 0.10
    f        = [0.7, 0.9]
    gamma    = [0.2, 0.6, 1, 1.4, 1.8]
    for _f in f:
        for _g in gamma:
            statistics(filepath, nBDs, rel_unc, _f, _g)

