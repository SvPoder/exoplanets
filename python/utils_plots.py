import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

## Plotting functions for exercise 1
def FSE_f_gamma_ex1(filepath, nBDs, rel_unc, rank=100, PE="median"):
    # grid points
    f     = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = np.array([0.2, 0.6, 1, 1.4, 1.8])
    
    FSE_1 = []; FSE_2 = [] 
    for _f in f:
        for _g in gamma:
            true = [_f, _g]
            data=np.genfromtxt(filepath+("N%i_relunc%.2f/statistics_ex1_N%i_relunc%.2f_f%.1fgamma%.1f" 
                                            %(nBDs, rel_unc, nBDs, rel_unc, _f, _g)), 
                                            unpack=True)
            if PE=="median":
                pe = np.array((data[2], data[3]))
            else:
                sys.exit("Need to implement other point estimates")

            FSE_1.append(np.sqrt(1/rank*np.sum(np.power(pe[0] - true[0], 2)))/true[0])
            FSE_2.append(np.sqrt(1/rank*np.sum(np.power(pe[1] - true[1], 2)))/true[1])

    xi, yi = np.mgrid[0:1:(len(f)+1)*1j, 0:2:(len(gamma)+1)*1j]
    zi_1   = np.array(FSE_1).reshape(len(f), len(gamma))
    zi_2   = np.array(FSE_2).reshape(len(f), len(gamma))
    # return
    return xi, yi, zi_1, zi_2


def plot_FSE_grid_f_gamma_ex1(filepath, fig, axes, rank=100, PE="median", 
                              plot_f=True):
    """
    Plot FSE in (f, gamma) plane for ex. 1 and 3 different numbers
    of BDs in simulation (100, 1000, 10000) and 2 different levels of 
    uncertainty in Tobs (0.05, 0.10)
    """
    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)
    
    _nBDs = [100, 1000, 10000]
    _rel_unc = [0.05, 0.1]
    nBDs    = []
    rel_unc = []

    for rel in _rel_unc:
        for n in _nBDs:
            nBDs.append(n)
            rel_unc.append(rel)
        
    for i, ax in enumerate(axes.flat):
        
        xi, yi, zi_1, zi_2 = FSE_f_gamma_ex1(filepath, nBDs[i], rel_unc[i], 
                                              rank=rank, PE=PE)
        
        if plot_f==True:
            im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
        else:
            im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r")
        
        if i==0 or i==3:
            ax.set_ylabel(r"$\gamma$")

        ax.set_xlabel(r"$f$")
        
        text_box = AnchoredText(("N=%i, unc T=%i" %(nBDs[i], int(rel_unc[i]*100))) + "$\%$", 
                                frameon=True, loc=2, pad=0.2)
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("FSE", size=18.)
    # return
    return

def plot_1Dposterior_ex1(nBDs, rel_unc, f, gamma):
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    xvals0 = np.linspace(0, 1, 100)
    xvals1 = np.linspace(0, 2, 100)
    
    filepath = ("../results/bayesian/ex1/N%i_relunc%.2f/" %(nBDs, rel_unc))
        
    for i in range(100):
        _file   = open(filepath + ("posterior_ex1_N%i_relunc%.2f_f%.1fgamma%.1fv%i" 
                                   %(nBDs, rel_unc, f, gamma, i)), "rb") 
        samples = pickle.load(_file)
        kde   = gaussian_kde(samples.T[0])
        ax[0].plot(xvals0, kde(xvals0)/np.max(kde(xvals0)), color="purple", lw=2.5, 
                   alpha=0.3)
        kde   = gaussian_kde(samples.T[1])
        ax[1].plot(xvals1, kde(xvals1)/np.max(kde(xvals1)), color="purple", lw=2.5, 
                   alpha=0.3)
        #ax[0].axvline(np.percentile(samples, [50], axis=0)[0, 0], ls="-", color="k")
        #ax[1].axvline(np.percentile(samples, [50], axis=0)[0, 1], ls="-", color="k")

    ax[0].axvline(f, ls="--", lw=2.5, color="red")
    ax[1].axvline(gamma, ls="--", lw=2.5, color="red")
    
    ax[0].set_xlabel(r"$f$")
    ax[1].set_xlabel(r"$\gamma$")
    
    fig.savefig("../Figs/1Dposterior_ex1_N%i_relunc%.2f_f%.1fgamma%.1f.pdf" 
                %(nBDs, rel_unc, f, gamma))
    # return
    return

## Plotting functions for exercise 2
def FSE_f_gamma_rs_ex2(filepath, nBDs, rel_unc, rank=100, PE="median"):
    # grid points
    f     = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = np.array([0.2, 0.6, 1, 1.4, 1.8])
    rs    = 20.
    
    FSE_1 = []; FSE_2 = []; FSE_3 = []
    for _f in f:
        for _g in gamma:
            true = [_f, _g, rs]
            data = np.genfromtxt(filepath +("N%i_relunc%.2f/statistics_ex2_N%i_relunc%.2f_f%.1fgamma%.1f" 
                                            %(nBDs, rel_unc, nBDs, rel_unc, _f, _g)), unpack=True)
            if PE=="median":
                pe = np.array((data[3], data[4], data[5]))
            else:
                sys.exit("Need to implement other point estimates")

            FSE_1.append(np.sqrt(1/rank*np.sum(np.power(pe[0] - true[0], 2)))/true[0])
            FSE_2.append(np.sqrt(1/rank*np.sum(np.power(pe[1] - true[1], 2)))/true[1])
            FSE_3.append(np.sqrt(1/rank*np.sum(np.power(pe[2] - true[2], 2)))/true[2])

    xi, yi = np.mgrid[0:1:(len(f)+1)*1j, 0:2:(len(gamma)+1)*1j]
    zi_1   = np.array(FSE_1).reshape(len(f), len(gamma))
    zi_2   = np.array(FSE_2).reshape(len(f), len(gamma))
    zi_3   = np.array(FSE_3).reshape(len(f), len(gamma))
    # return
    return xi, yi, zi_1, zi_2, zi_3


def plot_FSE_grid_f_gamma_ex2(filepath, fig, axes, rank=100, PE="median", 
                              plot_f=True, plot_g=True):
    """
    Plot FSE in (f, gamma) plane for exercise 2 and 3 different numbers
    of BDs in simulation (100, 1000, 10000) and 2 different levels of 
    uncertainty in Tobs (0.05, 0.10)
    """
    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)
    
    _nBDs = [100, 1000, 10000]
    _rel_unc = [0.05, 0.1]
    nBDs    = []
    rel_unc = []

    for rel in _rel_unc:
        for n in _nBDs:
            nBDs.append(n)
            rel_unc.append(rel)
        
    for i, ax in enumerate(axes.flat):
        
        xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs_ex2(filepath, nBDs[i], rel_unc[i], 
                                                      rank=rank, PE=PE)
        
        if plot_f==True:
            im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
        elif plot_g==True:
            im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r")
        else:
            im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap="cividis_r")
        
        if i==0 or i==3:
            ax.set_ylabel(r"$\gamma$")

        ax.set_xlabel(r"$f$")
        
        text_box = AnchoredText(("N=%i, unc T=%i" %(nBDs[i], int(rel_unc[i]*100))) + "$\%$", 
                                frameon=True, loc=2, pad=0.2)
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("FSE", size=18.)
    # return
    return
