import sys
import numpy as np
import pickle

def statistics(filepath, ex, nBDs, rel_unc, relM, f, gamma, rs, rank=100, D=2):
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
    
    for i in range(rank):
        print(i+1)
        # load posterior + likelihood

        file_name = (filepath + ("N%irelT%.2frelM%.2f/posterior_" 
                    %(nBDs, rel_unc, relM))
                    + ex + 
                    ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                    %(nBDs, rel_unc, relM, f, gamma, rs, i+1)))
        samples   = pickle.load(open(file_name, "rb"))

        # calculate point estimates
        for j in range(D):
            mean[j][i]   = np.mean(samples[:, j])
            median[j][i] = np.percentile(samples[:, j], [50], axis=0)
            #TODO need to change # bins to see if results differ
            _n, _bins    = np.histogram(samples[:, j], bins=50)
            MAP[j][i]    = _bins[np.argmax(_n)]
    filepath = "/home/mariacst/exoplanets/results/statistics_"
    output = open(filepath + ex + ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                              %(nBDs, rel_unc, relM, f, gamma, rs)), 
                  "w")
    for i in range(rank):
        for j in range(D):
            output.write("%.4f  " %mean[j][i])
        for j in range(D):
            output.write("%.4f  " %median[j][i])
        for j in range(D):
            output.write("%.4f  " %MAP[j][i])
        output.write("\n")
    output.close()

    # return
    return


if __name__ == '__main__':
    #filepath = "/home/mariacst/Tmin/results/"
    filepath = "/hdfs/local/mariacst/exoplanets/results/final_round/all_unc/Tmin/"
    ex = sys.argv[1]
    N  = int(sys.argv[2])
    relT = float(sys.argv[3])
    relM = float(sys.argv[4])
    print(N)
    nBDs     = [N]
    rel_unc  = [relT]
    rel_M    = [relM]
    f        = 1.
    rs       = [20.]
    gamma    = [1.0]

    for N in nBDs:
        for rel in rel_unc:
            for relM in rel_M:
                print(N, rel, relM)
                for _rs in rs:
                    for _g in gamma:
                        print(_rs, _g)
                        statistics(filepath, ex, N, rel, relM, f, _g, _rs, 100, 3)


