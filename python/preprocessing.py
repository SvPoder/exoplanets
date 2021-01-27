import numpy as np
import pickle

def statistics(filepath, nBDs, rank=100, D=2):
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
        file_name = (filepath + ("posterior_game0_nBDs_%iv%i" %(nBDs, i)))
        samples   = pickle.load(open(file_name, "rb"))

        file_name = (filepath + ("likelihood_game0_nBDs%iv%i" %(nBDs, i)))
        like      = pickle.load(open(file_name, "rb"))
        # calculate point estimates
        for j in range(D):
            mean[j][i]   = np.mean(samples[:, j])
            median[j][i] = np.percentile(samples[:, j], [50], axis=0)
            #TODO need to change # bins to see if results differ
            _n, _bins    = np.histogram(samples[:, j], bins=50)
            MAP[j][i]    = _bins[np.argmax(_n)]
            ML[j][i]     = samples[:, j][np.argmax(like)]

    output = open((filepath + "statistics_game0_nBDs_%i.dat" %nBDs), 
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
    filepath     = "../results/bayesian/game0/f1gamma1/"
    #rel_unc_Tobs = [0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5]
    nBDs = [100000]#[1000, 5000, 20000, 50000]
    for n in nBDs:
        statistics(filepath, n)

