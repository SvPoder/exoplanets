import numpy as np
import pickle

def statistics(filepath, nBDs, rel_unc, relM, f, gamma, rs, rank=100, D=2):
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
        #print(i)
        # load posterior + likelihood

        file_name = (filepath + ("posterior_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                    %(nBDs, rel_unc, relM, f, gamma, rs, i)))
        samples   = pickle.load(open(file_name, "rb"))

        file_name = (filepath + ("likelihood_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                    %(nBDs, rel_unc, relM, f, gamma, rs, i)))
        like      = pickle.load(open(file_name, "rb"))

        # calculate point estimates
        for j in range(D):
            mean[j][i]   = np.mean(samples[:, j])
            median[j][i] = np.percentile(samples[:, j], [50], axis=0)
            #TODO need to change # bins to see if results differ
            _n, _bins    = np.histogram(samples[:, j], bins=50)
            MAP[j][i]    = _bins[np.argmax(_n)]
            ML[j][i]     = samples[:, j][np.argmax(like)]

    output = open(filepath + ("statistics_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                              %(nBDs, rel_unc, relM, f, gamma, rs)), 
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
    filepath = "/Users/mariabenito/Desktop/results/"
    nBDs     = [10000]
    rel_unc  = [0.1]
    rel_M    = [0.1]
    f        = 1.
    rs       = [5., 10., 20.]
    gamma    = [0., 0.5, 1., 1.3, 1.5]

    for N in nBDs:
        for rel in rel_unc:
            for relM in rel_M:
                print(N, rel, relM)
                for _rs in rs:
                    for _g in gamma:
                        print(_rs, _g)
                        statistics(filepath, N, rel, relM, f, _g, _rs, 100, 3)

