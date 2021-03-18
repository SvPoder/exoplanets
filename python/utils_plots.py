import sys
import pickle
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from matplotlib import rc
rc('font', family='times new roman', size=22.)

# -----------------------------------
## Sensitivity
# -----------------------------------
def display_values(XX, YY, H, ax=False):
    if ax:
        for i in range(YY.shape[0]-1):
            for j in range(XX.shape[1]-1):
                ax.text((XX[i+1][0] + XX[i][0])/2, (YY[0][j+1] + YY[0][j])/2, '%i' % H[i, j],
                     horizontalalignment='center', verticalalignment='center', size=18)
    else:
        for i in range(YY.shape[0]-1):
            for j in range(XX.shape[1]-1):
                plt.text((XX[i+1][0] + XX[i][0])/2, (YY[0][j+1] + YY[0][j])/2, '%i' % H[i, j],
                     horizontalalignment='center', verticalalignment='center', size=18)
    # return
    return

def sensitivity_grid_ex1_f(filepath, nBDs, rel_unc, 
                     ax=False, show_bin_values=True):
    """
    Plot # of H0 acceptance out of rank in (f, gamma) plane
    """
    # grid points
    f     = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = np.array([0.2, 0.6, 1, 1.4, 1.8])

    zi = np.genfromtxt(filepath + ("N%i_relunc%.2f/sensitivity_ex1_N%i_relunc%.2f" 
                                            %(nBDs, rel_unc, nBDs, rel_unc)))   
    xi, yi = np.mgrid[0:1:(len(f)+1)*1j, 0:2:(len(gamma)+1)*1j]
    
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_ylabel(r"$\gamma$"); ax.set_xlabel(r"$f$")
    
    norm = colors.BoundaryNorm(boundaries=np.array([0, 5, 100]), ncolors=2)
    cmap = colors.ListedColormap(["#3F5F5F", "#FFFF66"])
    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap, edgecolor="black")
    
    if show_bin_values:
        display_values(xi, yi, zi, ax=ax)
    # return
    return


def sensitivity_grid_f(filepath, nBDs, rel_unc, relM,
                     ax=False, show_bin_values=True):
    """
    Plot # of H0 acceptance out of rank in (f, gamma) plane
    """
    # grid points
    f     = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = np.array([0.2, 0.6, 1, 1.4, 1.8])

    zi = np.genfromtxt(filepath + ("sensitivity_ex5_N%i_relunc%.2f_relM%.2f"
                                    %(nBDs, rel_unc, relM)))
    xi, yi = np.mgrid[0:1:(len(f)+1)*1j, 0:2:(len(gamma)+1)*1j]
    
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_ylabel(r"$\gamma$"); ax.set_xlabel(r"$f$")
    
    norm = colors.BoundaryNorm(boundaries=np.array([0, 5, 100]), ncolors=2)
    cmap = colors.ListedColormap(["#3F5F5F", "#FFFF66"])
    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap, edgecolor="black")
    
    if show_bin_values:
        display_values(xi, yi, zi, ax=ax)
    # return
    return
def grid_sensitivity(filepath, nBDs, rel_unc, relM, ex="ex3",
                     ax=False, y_label=True, x_label=True,
                     show_bin_values=True):
    """
    Plot # of H0 acceptance out of rank in (rs, gamma) plane
    """
    # grid points
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.2, 1.4])

    zi = np.genfromtxt(filepath + "sensitivity_" + ex +
                       ("_N%i_relunc%.2f_relM%.2f" %(nBDs, rel_unc, relM)))

    #print(zi.shape)
    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0., 0.25, 0.75, 1.05, 1.15, 1.25, 1.35,  1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if y_label==True:
        ax.set_ylabel(r"$\gamma$");
        ax.set_yticks(gamma)
        ax.set_yticklabels(['0', '0.5', '1', '1.2', '1.4'])
    else:
        ax.set_yticks(gamma)
        ax.set_yticklabels([])
    if x_label==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticks(rs)
        ax.set_xticklabels(['5', '10', '20'])
    else:
        ax.set_xticks(rs)
        ax.set_xticklabels([])

    norm = colors.BoundaryNorm(boundaries=np.array([0, 5, 100]), ncolors=2)
    cmap = colors.ListedColormap(["#3F5F5F", "#FFFF66"])
    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap, edgecolor="black")

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.)

    text_box = AnchoredText((r"$N=10^{%i}$"
                            %int(np.log10(nBDs))),
                            bbox_to_anchor=(0., 0.99),
                            bbox_transform=ax.transAxes, frameon=False,
                            pad=0., loc="lower left", prop=dict(size=19))

    ax.add_artist(text_box)

    if show_bin_values:
        display_values(xi, yi, zi, ax=ax)
    # return
    return


def grid_sensitivity_coarse(filepath, nBDs, rel_unc, relM, ex="ex3",
                            ax=False, y_label=True, x_label=True, 
                            show_bin_values=True):
    """
    Plot # of H0 acceptance out of rank in (rs, gamma) plane
    """
    # grid points
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.3, 1.5])

    zi = np.genfromtxt(filepath + "sensitivity_" + ex +
                       ("_N%i_relunc%.2f_relM%.2f" %(nBDs, rel_unc, relM)))

    #print(zi.shape)
    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0., 0.25, 0.75, 1.15, 1.4, 1.6])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")
    
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if y_label==True:
        ax.set_ylabel(r"$\gamma$"); 
        ax.set_yticks(gamma)
        ax.set_yticklabels(['0', '0.5', '1', '1.3', '1.5'])
    else:
        ax.set_yticks(gamma)
        ax.set_yticklabels([])
    if x_label==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticks(rs)
        ax.set_xticklabels(['5', '10', '20'])
    else:
        ax.set_xticks(rs)
        ax.set_xticklabels([])
    
    norm = colors.BoundaryNorm(boundaries=np.array([0, 5, 100]), ncolors=2)
    cmap = colors.ListedColormap(["#3F5F5F", "#FFFF66"])
    ax.pcolormesh(xi, yi, zi, norm=norm, cmap=cmap, edgecolor="black")

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.)

    text_box = AnchoredText((r"$N=10^{%i}$"
                            %int(np.log10(nBDs))),
                            bbox_to_anchor=(0., 0.99),
                            bbox_transform=ax.transAxes, frameon=False, 
                            pad=0., loc="lower left", prop=dict(size=19))
        
    ax.add_artist(text_box)

    if show_bin_values:
        display_values(xi, yi, zi, ax=ax)
    # return
    return


# -----------------------------------
## FSE
# -----------------------------------
def add_hatch(ax, i, j, width, height):
    ax.add_patch(mpatches.Rectangle(
              (i, j),
              width,
              height, 
              fill=False, 
              color='yellow', linewidth=0.,
              hatch='//')) # the more slashes, the denser the hash lines 
    return


def FSE_f_gamma_rs_coarse(filepath, nBDs, rel_unc, relM, ex, rank=100, PE="median"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.3, 1.5])

    FSE_1 = []; FSE_2 = []; FSE_3 = []
    for _rs in rs:
        for _g in gamma:
            true = [f, _g, _rs]
            data = np.genfromtxt(filepath + "statistics_" + ex +
                                 ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f"
                                  %(nBDs, rel_unc, relM, f, _g, _rs)), unpack=True)
            if PE=="median":
                pe = np.array((data[3], data[4], data[5]))
            else:
                sys.exit("Need to implement other point estimates")
            FSE_1.append(np.sqrt(1/rank*np.sum(np.power(pe[0] - true[0], 2)))/true[0])
            if np.abs(_g) < 1e-5:
                epsilon=1e-4
            else:
                epsilon=0.
            FSE_2.append(np.sqrt(1/rank*np.sum(np.power(pe[1] - true[1], 2)))/(true[1]+epsilon))
            FSE_3.append(np.sqrt(1/rank*np.sum(np.power(pe[2] - true[2], 2)))/true[2])

    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0., 0.25, 0.75, 1.15, 1.4, 1.6])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    zi_1   = np.array(FSE_1).reshape(len(rs), len(gamma))
    zi_2   = np.array(FSE_2).reshape(len(rs), len(gamma))
    zi_3   = np.array(FSE_3).reshape(len(rs), len(gamma))
    # return
    return xi, yi, zi_1, zi_2, zi_3

def FSE_f_gamma_rs(filepath, nBDs, rel_unc, relM, ex, rank=100, PE="median"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    FSE_1 = []; FSE_2 = []; FSE_3 = []
    for _rs in rs:
        for _g in gamma:
            true = [f, _g, _rs]
            data = np.genfromtxt(filepath + "statistics_" + ex + 
                                 ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1f" 
                                  %(nBDs, rel_unc, relM, f, _g, _rs)), unpack=True)
            if PE=="median":
                pe = np.array((data[3], data[4], data[5]))
            else:
                sys.exit("Need to implement other point estimates")
            FSE_1.append(np.sqrt(1/rank*np.sum(np.power(pe[0] - true[0], 2)))/true[0])
            if np.abs(_g) < 1e-5:
                epsilon=1e-4
            else:
                epsilon=0.
            FSE_2.append(np.sqrt(1/rank*np.sum(np.power(pe[1] - true[1], 2)))/(true[1]+epsilon))
            FSE_3.append(np.sqrt(1/rank*np.sum(np.power(pe[2] - true[2], 2)))/true[2])

    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0., 0.25, 0.75, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    zi_1   = np.array(FSE_1).reshape(len(rs), len(gamma))
    zi_2   = np.array(FSE_2).reshape(len(rs), len(gamma))
    zi_3   = np.array(FSE_3).reshape(len(rs), len(gamma))
    # return
    return xi, yi, zi_1, zi_2, zi_3

def grid_FSE(filepath, nBDs, rel_unc, relM, ex="ex3",
             ax=False, PE="median",
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,
             rank=100):
    """
    Plot FSE grid in (rs, gamma) 
    """

    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)

    xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs(filepath, nBDs, rel_unc, relM,
                                              ex, rank=rank, PE=PE)
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if plot_f==True:
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
    elif plot_g==True:
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r")
    else:
        im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap="cividis_r")
    if ylabel==True:
        ax.set_ylabel(r"$\gamma$")
        ax.set_yticklabels(['0', '0.5', '1', '', '1.2', '', '1.4', ''])
    else:
        ax.set_yticklabels([])
    if xlabel==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticklabels(['5', '10', '20'])
    else:
        ax.set_xticklabels([])

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_T$=%i"
                            %(int(np.log10(nBDs)), int(rel_unc*100))
                            + "$\%, $"
                            + "$\sigma_M$=%i" %(int(relM*100)) + "$\%$"),
                            frameon=True, loc=3, pad=0.2, prop=dict(size=20))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.set_xticks([5., 10., 20.])
    ax.set_yticks([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im


def grid_FSE_coarse(filepath, nBDs, rel_unc, relM, ex="ex3", 
             ax=False, PE="median", 
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,
             rank=100):
    """
    Plot FSE grid in (rs, gamma) 
    """

    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)
    
    xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs_coarse(filepath, nBDs, rel_unc, relM,
                                              ex, rank=rank, PE=PE)
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if plot_f==True:
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
    elif plot_g==True:
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r")
    else:
        im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap="cividis_r")
    if ylabel==True:
        ax.set_ylabel(r"$\gamma$")
        ax.set_yticklabels(['0', '0.5', '1', '1.3', '1.5'])
    else:
        ax.set_yticklabels([])
    if xlabel==True:
        ax.set_xlabel(r"$r_s$ [kpc]")
        ax.set_xticklabels(['5', '10', '20'])
    else:
        ax.set_xticklabels([])

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_T$=%i" 
                            %(int(np.log10(nBDs)), int(rel_unc*100)) 
                            + "$\%, $" 
                            + "$\sigma_M$=%i" %(int(relM*100)) + "$\%$"), 
                            frameon=True, loc=3, pad=0.2, prop=dict(size=20))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.set_xticks([5., 10., 20.])
    ax.set_yticks([0., 0.5, 1, 1.3, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im


# -----------------------------------
## Posterior
# -----------------------------------

def plot_1Dposterior(filepath, nBDs, rel_unc, relM, ex,
                     f, gamma, rs, color="k"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    xvals = [np.linspace(0, 1, 100), np.linspace(0, 3, 100), 
             np.linspace(0, 50, 100)]

    true = [f, gamma, rs]
    
    filepath = filepath + ("N%irelT%.2frelM%.2f/" %(nBDs, rel_unc, relM))

    for i, ax in enumerate(axes.flat):
        print("i = ", i)
        for j in range(100):
            _file   = open(filepath + "posterior_" + ex + 
                           ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                           %(nBDs, rel_unc, relM, f, gamma, rs, j+1)), "rb") 
            samples = pickle.load(_file)
            kde   = gaussian_kde(samples.T[i])
            ax.plot(xvals[i], kde(xvals[i])/np.max(kde(xvals[i])), 
                    color=color, lw=2.5, 
                    alpha=0.3)
        ax.axvline(true[i], ls="--", lw=2.5, color="red")
        if i==0:
            ax.set_xlabel(r"$f$")
            ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
            text_box = AnchoredText((r"N=%i, $\sigma_T$=%i" %(nBDs, int(rel_unc*100)) 
                                + "$\%, $" 
                                + "$\sigma_M$=%i" %(int(relM*100)) + "$\%$"),
                                bbox_to_anchor=(0., 0.99),
                                bbox_transform=ax.transAxes, loc='lower left', 
                                pad=0.04, prop=dict(size=20))
            plt.setp(text_box.patch, facecolor="white")
            ax.add_artist(text_box)
        elif i==1:
            ax.set_xlabel(r"$\gamma$")
            ax.set_xticks([0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.])
            ax.set_xticklabels(['0.2', '0.6', '1', '1.4', '1.8', '2.2', '2.6', '3'])
        else:
            ax.set_xlabel(r"$r_s$ [kpc]")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.)
    
    fig.subplots_adjust(hspace=0.25, wspace=0.08)
    fig.savefig("../../Figs/1Dposterior_" + ex + 
                ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%i.pdf" 
                %(nBDs, rel_unc, relM, f, gamma, int(rs))), bbox_inches="tight")
    # return
    return

# -----------------------------------
## Plotting functions for exercise 1
# -----------------------------------
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
        if i>=3:
            ax.set_xlabel(r"$f$")
        
        text_box = AnchoredText(("N=%i, unc T=%i" %(nBDs[i], int(rel_unc[i]*100))) + "$\%$", 
                                frameon=True, loc=2, pad=0.2)
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)

        plt.yticks([0.2, 0.6, 1.0, 1.4, 1.8], ['0.2', '0.6', '1', '1.4', '1.8'])
        plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], ['0.1', '0.3', '0.5', '0.7', '0.9'])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)

    fig.subplots_adjust(wspace=0.08, hspace=0.08)
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


def plot_1Dposterior_ex1(f, gamma, rs, dir_path, plot_f=True, color="k"):
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)
    
    if plot_f==True:
        xvals = np.linspace(0, 1, 100)
        k = 0
    else:
        xvals = np.linspace(0, 2, 100)
        k = 1

    true = [f, gamma]

    N   = [100, 1000, 10000]
    unc = [0.05, 0.10]
    nBDs = []; rel_unc = []
    for r in unc:
        for n in N:
            nBDs.append(n)
            rel_unc.append(r)
        
    for i, ax in enumerate(axes.flat):
        
        filepath = (dir_path + "N%i_relunc%.2f/" %(nBDs[i], rel_unc[i]))
        for j in range(100):
            _file   = open(filepath + ("posterior_ex1_N%i_relunc%.2f_f%.1fgamma%.1fv%i" 
                                   %(nBDs[i], rel_unc[i], f, gamma, j)), "rb") 
            samples = pickle.load(_file)
            kde   = gaussian_kde(samples.T[k])
            ax.plot(xvals, kde(xvals)/np.max(kde(xvals)), color=color, lw=2.5, 
                    alpha=0.3)

        ax.axvline(true[k], ls="--", lw=2.5, color="red")
        if i>=3:
            if k==0:
                ax.set_xlabel(r"$f$")
                plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], 
                           ['0.1', '0.3', '0.5', '0.7', '0.9'])
            else:
                ax.set_xlabel(r"$\gamma$")
                plt.xticks([0.2, 0.6, 1.0, 1.4, 1.8], 
                           ['0.2', '0.6', '1', '1.4', '1.8'])

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.)

        text_box = AnchoredText(("N=%i, unc T=%i" %(nBDs[i], int(rel_unc[i]*100))) + "$\%$", 
                                bbox_to_anchor=(0., 0.99),
                                bbox_transform=ax.transAxes, loc='lower left',
                                pad=0.04, prop=dict(size=20))
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)
    
    fig.subplots_adjust(hspace=0.2, wspace=0.08)
    fig.savefig("../Figs/1Dposterior_ex1_f%.1fgamma%.1frs%i_%i.pdf" 
                %(f, gamma, int(rs), k))
    # return
    return


# -----------------------------------
## Plotting functions for exercise 2
# -----------------------------------
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
        if i>=3:
            ax.set_xlabel(r"$f$")
        
        text_box = AnchoredText(("N=%i, unc T=%i" %(nBDs[i], int(rel_unc[i]*100))) + "$\%$", 
                                frameon=True, loc=2, pad=0.2)
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)

        plt.yticks([0.2, 0.6, 1.0, 1.4, 1.8], ['0.2', '0.6', '1', '1.4', '1.8'])
        plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], ['0.1', '0.3', '0.5', '0.7', '0.9'])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)

    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(right=0.89)
    cbar_ax = fig.add_axes([0.91, 0.25, 0.02, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("FSE", size=18.)
    # return
    return

def plot_1Dposterior_ex2_bis(nBDs, rel_unc, f, gamma, rs=20., color="black"):
    
    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    
    xvals0 = np.linspace(0, 1, 100)
    xvals1 = np.linspace(0, 2, 100)
    xvals2 = np.linspace(0, 50, 100)
    
    filepath = ("/Users/mariabenito/Desktop/results/ex2/N%i_relunc%.2f/" %(nBDs, rel_unc))
        
    for i in range(100):
        _file   = open(filepath + ("posterior_ex2_N%i_relunc%.2f_f%.1fgamma%.1fv%i" 
                                   %(nBDs, rel_unc, f, gamma, i)), "rb") 
        samples = pickle.load(_file)
        kde   = gaussian_kde(samples.T[0])
        ax[0].plot(xvals0, kde(xvals0)/np.max(kde(xvals0)), color=color, lw=2.5, 
                   alpha=0.3)
        kde   = gaussian_kde(samples.T[1])
        ax[1].plot(xvals1, kde(xvals1)/np.max(kde(xvals1)), color=color, lw=2.5, 
                   alpha=0.3)
        kde   = gaussian_kde(samples.T[2])
        ax[2].plot(xvals2, kde(xvals2)/np.max(kde(xvals2)), color=color, lw=2.5, 
                   alpha=0.3)
        #ax[0].axvline(np.percentile(samples, [50], axis=0)[0, 0], ls="-", color="k")
        #ax[1].axvline(np.percentile(samples, [50], axis=0)[0, 1], ls="-", color="k")

    ax[0].axvline(f, ls="--", lw=2.5, color="red")
    ax[1].axvline(gamma, ls="--", lw=2.5, color="red")
    ax[2].axvline(rs, ls="--", lw=2.5, color="red")
    
    ax[0].set_xlabel(r"$f$")
    ax[1].set_xlabel(r"$\gamma$")
    ax[2].set_xlabel(r"$r_s$ [kpc]")
    
    fig.savefig("../Figs/1Dposterior_ex2_N%i_relunc%.2f_f%.1fgamma%.1f.pdf" 
                %(nBDs, rel_unc, f, gamma))
    # return
    return

def plot_1Dposterior_ex2(f, gamma, rs, dir_path, plot_f=True, plot_g=False, 
                         color="k"):
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10), sharex=True, sharey=True)
    
    if plot_f==True and plot_g==False:
        xvals = np.linspace(0, 1, 100)
        k = 0
    elif plot_f==False and plot_g==True:
        xvals = np.linspace(0, 2, 100)
        k = 1
    else:
        xvals = np.linspace(0, 50, 100)
        k=2

    true = [f, gamma, rs]

    N   = [100, 1000, 10000]
    unc = [0.05, 0.10]
    nBDs = []; rel_unc = []
    for r in unc:
        for n in N:
            nBDs.append(n)
            rel_unc.append(r)
        
    for i, ax in enumerate(axes.flat):
        
        filepath = (dir_path + "N%i_relunc%.2f/" %(nBDs[i], rel_unc[i]))
        for j in range(100):
            _file   = open(filepath + ("posterior_ex2_N%i_relunc%.2f_f%.1fgamma%.1fv%i" 
                                   %(nBDs[i], rel_unc[i], f, gamma, j)), "rb") 
            samples = pickle.load(_file)
            kde   = gaussian_kde(samples.T[k])
            ax.plot(xvals, kde(xvals)/np.max(kde(xvals)), color=color, lw=2.5, 
                    alpha=0.3)

        ax.axvline(true[k], ls="--", lw=2.5, color="red")
        if i>=3:
            if k==0:
                ax.set_xlabel(r"$f$")
                plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], 
                           ['0.1', '0.3', '0.5', '0.7', '0.9'])
            elif k==1:
                ax.set_xlabel(r"$\gamma$")
                plt.xticks([0.2, 0.6, 1.0, 1.4, 1.8], 
                           ['0.2', '0.6', '1', '1.4', '1.8'])
            else:
                ax.set_xlabel(r"$r_s$ [kpc]")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.)

        text_box = AnchoredText(("N=%i, unc T=%i" %(nBDs[i], int(rel_unc[i]*100))) + "$\%$", 
                                bbox_to_anchor=(0., 0.99),
                                bbox_transform=ax.transAxes, loc='lower left',
                                pad=0.04, prop=dict(size=20))
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)
    
    fig.subplots_adjust(hspace=0.2, wspace=0.08)
    fig.savefig("../Figs/1Dposterior_ex2_f%.1fgamma%.1frs%i_%i.pdf" 
                %(f, gamma, int(rs), k))
    # return
    return


# -----------------------------------
## Plotting functions for exercise 3
# -----------------------------------
def FSE_f_gamma_rs_ex3(filepath, nBDs, rel_unc, relM, rank=100, PE="median"):
    # grid points
    f     = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = np.array([0.2, 0.6, 1, 1.4, 1.8])
    rs    = 20.
    
    FSE_1 = []; FSE_2 = []; FSE_3 = []
    for _f in f:
        for _g in gamma:
            true = [_f, _g, rs]
            data = np.genfromtxt(filepath + 
                ("N%i_relunc%.2f/statistics_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1f" 
                %(nBDs, rel_unc, nBDs, rel_unc, relM, _f, _g)), unpack=True)
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

def plot_FSE_grid_f_gamma_ex3(filepath, fig, axes, rank=100, PE="median", 
                              plot_f=True, plot_g=True):
    """
    Plot FSE in (f, gamma) plane for exercise 2 and 3 different numbers
    of BDs in simulation (100, 1000, 10000) and 2 different levels of 
    uncertainty in Tobs (0.05, 0.10)
    """
    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)
    
    _nBDs    = [100, 10000]
    _rel_unc = [0.05, 0.1]
    _relM    = [0.10, 0.20]
    nBDs     = []
    rel_unc  = []
    relM     = []
    
    for n in _nBDs:
        for rel in _rel_unc:
            for rM in _relM:
                nBDs.append(n)
                rel_unc.append(rel)
                relM.append(rM)
        
    for i, ax in enumerate(axes.flat):
        
        xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs_ex3(filepath, nBDs[i], rel_unc[i], 
                                                      relM[i], rank=rank, PE=PE)
        
        if plot_f==True:
            im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
        elif plot_g==True:
            im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r")
        else:
            im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap="cividis_r")
        
        if i==0 or i==4:
            ax.set_ylabel(r"$\gamma$")
        if i>=4:
            ax.set_xlabel(r"$f$")
        
        text_box = AnchoredText((r"N=%i, $\sigma_T$=%i" %(nBDs[i], int(rel_unc[i]*100)) + "$\%, $" 
                                + "$\sigma_M$=%i" %(int(relM[i]*100)) + "$\%$"), 
                                frameon=True, loc=2, pad=0.2, prop=dict(size=19))
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)

        plt.yticks([0.2, 0.6, 1.0, 1.4, 1.8], ['0.2', '0.6', '1', '1.4', '1.8'])
        plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], ['0.1', '0.3', '0.5', '0.7', '0.9'])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)

    fig.subplots_adjust(wspace=0.06, hspace=0.06)

    fig.subplots_adjust(right=0.91)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.4])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("FSE", size=18.)
    # return
    return


def plot_1Dposterior_ex3(f, gamma, rs, dir_path, plot_f=True, plot_g=False, 
                         color="k"):
    
    fig, axes = plt.subplots(2, 4, figsize=(19, 10), sharex=True, sharey=True)
    
    if plot_f==True and plot_g==False:
        xvals = np.linspace(0, 1, 100)
        k = 0
    elif plot_f==False and plot_g==True:
        xvals = np.linspace(0, 2, 100)
        k = 1
    else:
        xvals = np.linspace(0, 50, 100)
        k=2

    true = [f, gamma, rs]

    _nBDs    = [100, 10000]
    _rel_unc = [0.05, 0.1]
    _relM    = [0.10, 0.20]
    nBDs     = []
    rel_unc  = []
    relM     = []
    
    for n in _nBDs:
        for rel in _rel_unc:
            for rM in _relM:
                nBDs.append(n)
                rel_unc.append(rel)
                relM.append(rM)
        
    for i, ax in enumerate(axes.flat):
        
        filepath = (dir_path + "N%i_relunc%.2f/" %(nBDs[i], rel_unc[i]))
        for j in range(100):
            _file   = open(filepath + ("posterior_ex3_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                           %(nBDs[i], rel_unc[i], relM[i], f, gamma, rs, j)), "rb") 
            samples = pickle.load(_file)
            kde   = gaussian_kde(samples.T[k])
            ax.plot(xvals, kde(xvals)/np.max(kde(xvals)), color=color, lw=2.5, 
                    alpha=0.3)

        ax.axvline(true[k], ls="--", lw=2.5, color="red")
        if i>=4:
            if k==0:
                ax.set_xlabel(r"$f$")
                plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], 
                           ['0.1', '0.3', '0.5', '0.7', '0.9'])
            elif k==1:
                ax.set_xlabel(r"$\gamma$")
                plt.xticks([0.2, 0.6, 1.0, 1.4, 1.8], 
                           ['0.2', '0.6', '1', '1.4', '1.8'])
            else:
                ax.set_xlabel(r"$r_s$ [kpc]")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.)

        text_box = AnchoredText((r"N=%i, $\sigma_T$=%i" %(nBDs[i], int(rel_unc[i]*100)) + "$\%, $" 
                                + "$\sigma_M$=%i" %(int(relM[i]*100)) + "$\%$"),
                                bbox_to_anchor=(0., 0.99),
                                bbox_transform=ax.transAxes, loc='lower left', pad=0.04, prop=dict(size=20))
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)
    
    fig.subplots_adjust(hspace=0.2, wspace=0.08)
    fig.savefig("../Figs/1Dposterior_ex3_f%.1fgamma%.1frs%i_%i.pdf" 
                %(f, gamma, int(rs), k))
    # return
    return


# --------------------------------------
## Plotting functions for exercise 4 & 5
# --------------------------------------
def FSE_f_gamma_rs_ex(filepath, nBDs, rel_unc, relM, ex, rank=100, PE="median"):
    # grid points
    f     = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    gamma = np.array([0.2, 0.6, 1, 1.4, 1.8])
    rs    = 20.
    
    FSE_1 = []; FSE_2 = []; FSE_3 = []
    for _f in f:
        for _g in gamma:
            true = [_f, _g, rs]
            data = np.genfromtxt(filepath + "statistics_" + ex + 
                                 ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1f" 
                                  %(nBDs, rel_unc, relM, _f, _g)), unpack=True)
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


def plot_FSE_grid_f_gamma_ex(filepath, fig, axes, ex, rank=100, PE="median", 
                             plot_f=True, plot_g=True):
    """
    Plot FSE in (f, gamma) plane for exercise 2 and 3 different numbers
    of BDs in simulation (100, 1000, 10000) and 2 different levels of 
    uncertainty in Tobs (0.05, 0.10)
    """

    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)
    
    _nBDs    = [100, 1000]
    _rel_unc = [0.1]
    _relM    = [0.1]
    nBDs     = []
    rel_unc  = []
    relM     = []
    
    for n in _nBDs:
        for rel in _rel_unc:
            for rM in _relM:
                nBDs.append(n)
                rel_unc.append(rel)
                relM.append(rM)

    for i, ax in enumerate(axes.flat):
        
        xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs_ex(filepath, nBDs[i], rel_unc[i], 
                                                     relM[i],
                                                     ex, rank=rank, PE=PE)
        
        if plot_f==True:
            im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
        elif plot_g==True:
            im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r")
        else:
            im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap="cividis_r")
        
        if i==0 or i==3:
            ax.set_ylabel(r"$\gamma$")
        if i>=3:
            ax.set_xlabel(r"$f$")
        
        text_box = AnchoredText((r"N=%i, $\sigma_T$=%i" %(nBDs[i], int(rel_unc[i]*100)) + "$\%, $" 
                                + "$\sigma_M$=%i" %(int(relM[i]*100)) + "$\%$"), 
                                frameon=True, loc=2, pad=0.2, prop=dict(size=20))
        plt.setp(text_box.patch, facecolor="white")
        ax.add_artist(text_box)

        plt.yticks([0.2, 0.6, 1.0, 1.4, 1.8], ['0.2', '0.6', '1', '1.4', '1.8'])
        plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9], ['0.1', '0.3', '0.5', '0.7', '0.9'])
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.5)

    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.subplots_adjust(right=0.89)
    cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title("FSE", size=18.)
    # return
    return


def plot_1Dposterior_ex(f, gamma, rs, filepath, ex, color="k"):
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 16), sharey=True)
    
    xvals = [np.linspace(0, 1, 100), np.linspace(0, 2, 100), 
             np.linspace(0, 50, 100)]

    nBDs    = [100, 1000, 100, 1000, 100, 1000]
    rel_unc = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    relM    = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    k = [0, 0, 1, 1, 2, 2]
    true = [f, f, gamma, gamma, rs, rs]

    for i, ax in enumerate(axes.flat):
        for j in range(100):
            _file   = open(filepath + "posterior_" + ex + 
                           ("_N%i_relunc%.2f_relM%.2f_f%.1fgamma%.1frs%.1fv%i" 
                           %(nBDs[i], rel_unc[i], relM[i], f, gamma, rs, j)), "rb") 
            samples = pickle.load(_file)
            kde   = gaussian_kde(samples.T[k[i]])
            ax.plot(xvals[k[i]], kde(xvals[k[i]])/np.max(kde(xvals[k[i]])), 
                    color=color, lw=2.5, 
                    alpha=0.3)

        ax.axvline(true[i], ls="--", lw=2.5, color="red")
        if i==0 or i==1:
            ax.set_xlabel(r"$f$")
            ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.set_xticklabels(['0.1', '0.3', '0.5', '0.7', '0.9'])
            text_box = AnchoredText((r"N=%i, $\sigma_T$=%i" %(nBDs[i], int(rel_unc[i]*100)) + "$\%, $" 
                                + "$\sigma_M$=%i" %(int(relM[i]*100)) + "$\%$"),
                                bbox_to_anchor=(0., 0.99),
                                bbox_transform=ax.transAxes, loc='lower left', pad=0.04, prop=dict(size=20))
            plt.setp(text_box.patch, facecolor="white")
            ax.add_artist(text_box)

        elif i==2 or i==3:
            ax.set_xlabel(r"$\gamma$")
            ax.set_xticks([0.2, 0.6, 1.0, 1.4, 1.8])
            ax.set_xticklabels(['0.2', '0.6', '1', '1.4', '1.8'])
        else:
            ax.set_xlabel(r"$r_s$ [kpc]")

        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2.)
    
    fig.subplots_adjust(hspace=0.25, wspace=0.08)
    fig.savefig("../Figs/1Dposterior_" + ex + ("_f%.1fgamma%.1frs%i.pdf" 
                %(f, gamma, int(rs))))
    # return
    return
