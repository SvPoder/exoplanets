from matplotlib.lines import Line2D
from _corner import corner
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
            elif PE=="ML":
                pe = np.array((data[15], data[16], data[17]))
            else:
                sys.exit("Point estimate not implemented!")
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

def FSE_f_gamma_rs(filepath, nBDs, rel_unc, ex, rank=100, PE="median"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    FSE_1 = []; FSE_2 = []; FSE_3 = []
    for _rs in rs:
        for _g in gamma:
            true = [f, _g, _rs]
            data = np.genfromtxt(filepath + "statistics_" + ex + 
                                 ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f" 
                                  %(nBDs, rel_unc, f, _g, _rs)), unpack=True)
            if PE=="median":
                pe = np.array((data[3], data[4], data[5]))
            elif PE=="mode":
                pe = np.array((data[12], data[13], data[14]))
            elif PE=="mean":
                pe = np.array((data[0], data[1], data[2]))
            elif PE=="ML":
                pe = np.array((data[15], data[16], data[17]))
            else:
                sys.exit("Point estimate not implemented!")
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


                                                                                    
def MSE_f_gamma_rs(filepath, nBDs, rel_unc, ex, rank=100, PE="median"):             
    # grid points                                                                   
    f     = 1.                                                                      
    rs    = np.array([5., 10., 20.])                                                
    gamma = np.array([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])                         
                                                                                    
    MSE_1 = []; MSE_2 = []; MSE_3 = []                                              
    for _rs in rs:                                                                  
        for _g in gamma:                                                            
            true = [f, _g, _rs]                                                     
            data = np.genfromtxt(filepath + "statistics_" + ex +                    
                                 ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f"             
                                  %(nBDs, rel_unc, f, _g, _rs)), unpack=True)       
            if PE=="median":                                                    
                pe = np.array((data[3], data[4], data[5]))                          
            elif PE=="mode":                                                        
                pe = np.array((data[12], data[13], data[14]))                       
            elif PE=="mean":                                                        
                pe = np.array((data[0], data[1], data[2]))                          
            elif PE=="ML":                                                          
                pe = np.array((data[15], data[16], data[17]))                       
            else:                                                                   
                sys.exit("Point estimate not implemented!")                         
            MSE_1.append(1/rank*np.sum(np.power(pe[0] - true[0], 2)))
            if np.abs(_g) < 1e-5:                                               
                epsilon=1e-4                                                        
            else:                                                                   
                epsilon=0.                                                          
            MSE_2.append(1/rank*np.sum(np.power(pe[1] - true[1], 2)))
            MSE_3.append(1/rank*np.sum(np.power(pe[2] - true[2], 2)))
                                                                                
    xi = np.array([2.5, 7.5, 15, 25])                                           
    yi = np.array([0., 0.25, 0.75, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55])             
    xi, yi = np.meshgrid(xi, yi, indexing="ij")                                     
                                                                                    
    zi_1   = np.array(MSE_1).reshape(len(rs), len(gamma))                           
    zi_2   = np.array(MSE_2).reshape(len(rs), len(gamma))                           
    zi_3   = np.array(MSE_3).reshape(len(rs), len(gamma))                           
    # return                                                                        
    return xi, yi, zi_1, zi_2, zi_3, MSE_2   


def FSE_f_gamma_rs_each(filepath, nBDs, rel_unc, relM, relA, relR, ex, 
                        rank=100, PE="median"):   
    # grid points                                                               
    f     = 1.                                                                  
    rs    = np.array([5., 10., 20.])                                            
    gamma = np.array([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])                     
                                                                                
    FSE_1 = []; FSE_2 = []; FSE_3 = []                                          
    for _rs in rs:                                                              
        for _g in gamma:                                                        
            true = [f, _g, _rs]                                                 
            data = np.genfromtxt(filepath + "statistics_" + ex +                
                                 ("_N%i_relunc%.2f_relM%.2f_relA%.2f_relR%.2f_f%.1fgamma%.1frs%.1f"
                                  %(nBDs, rel_unc, relM, relA, relR, f, _g, _rs)), unpack=True)
            if PE=="median":
                pe = np.array((data[3], data[4], data[5]))
            elif PE=="ML":
                pe = np.array((data[15], data[16], data[17]))
            else:
                sys.exit("Point estimate not implemented!")
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
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.set_xticks([5., 10., 20.])
    ax.set_yticks([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im

def grid_FSE_all(filepath, nBDs, rel_unc, ex="ex1",
             ax=False, PE="median",
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,
             rank=100):
    """
    Plot FSE grid in (rs, gamma) 
    """

    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)

    xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs(filepath, nBDs, rel_unc,
                                              ex, rank=rank, PE=PE)
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if plot_f==True:
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")
    elif plot_g==True:
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r", linewidth=0)
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

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_i$=%i"
                            %(int(np.log10(nBDs)), int(rel_unc*100))
                            + "$\% $"),
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.set_xticks([5., 10., 20.])
    ax.set_yticks([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im

def grid_FSE_each(filepath, nBDs, rel_unc, relM, relA, relR, ex="ex17",                       
             ax=False, PE="median",                                             
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,             
             rank=100):                                                         
    """                                                                         
    Plot FSE grid in (rs, gamma)                                                
    """                                                                         
                                                                                
    norm = colors.BoundaryNorm(boundaries=np.arange(0, 1, 0.05), ncolors=256)   
                                                                                
    xi, yi, zi_1, zi_2, zi_3 = FSE_f_gamma_rs_each(filepath, nBDs, rel_unc, relM,
                                              relA, relR,    
                                              ex, rank=rank, PE=PE)             
    if ax==False:                                                               
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))                            
    if plot_f==True:                                                            
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap="magma_r")             
    elif plot_g==True:                                                          
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap="viridis_r", linewidth=0,)
        im.set_edgecolor("face")
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

    if np.abs(relM+relA+relR) < 1e-3:
        variable = "T"
        rel      = rel_unc
    elif np.abs(relA+relR+rel_unc) < 1e-3:
        variable = "M"
        rel      = relM
    elif np.abs(relR+rel_unc+relM) < 1e-3:
        variable = "A"
        rel      = relA
    else:
        variable = "R"
        rel      = relR
                                                                                
    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_%s$=%i"                      
                            %(int(np.log10(nBDs)), variable, int(rel*100))  
                            + "$\% $"),                                         
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))   
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
## Coverage
# -----------------------------------
def coverage_f_gamma_rs(filepath, nBDs, rel_unc, ex, rank=100, CR="symmetric"):
    # grid points
    f     = 1.
    rs    = np.array([5., 10., 20.])
    gamma = np.array([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    cove_1 = []; cove_2 = []; cove_3 = []
    for _rs in rs:
        for _g in gamma:
            true = [f, _g, _rs]
            data = np.genfromtxt(filepath + "statistics_" + ex + 
                                 ("_N%i_sigma%.1f_f%.1fgamma%.1frs%.1f" 
                                  %(nBDs, rel_unc, f, _g, _rs)), unpack=True)
            if CR=="symmetric":
                low  = np.array((data[6], data[7], data[8]))
                high = np.array((data[9], data[10], data[11]))
            else:
                sys.exit("Credible interval not implemented!")
            one = f > low[0]
            two = f < high[0]
            cove_1.append(len(np.where((one==True) & (two==True))[0]))
            one = _g > low[1]
            two = _g < high[1]
            cove_2.append(len(np.where((one==True) & (two==True))[0]))
            one = _rs > low[2]
            two = _rs < high[2]
            cove_3.append(len(np.where((one==True) & (two==True))[0]))
    xi = np.array([2.5, 7.5, 15, 25])
    yi = np.array([0., 0.25, 0.75, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55])
    xi, yi = np.meshgrid(xi, yi, indexing="ij")

    zi_1   = np.array(cove_1).reshape(len(rs), len(gamma))
    zi_2   = np.array(cove_2).reshape(len(rs), len(gamma))
    zi_3   = np.array(cove_3).reshape(len(rs), len(gamma))
    # return
    return xi, yi, zi_1, zi_2, zi_3


def coverage_f_gamma_rs_each(filepath, nBDs, rel_unc, relM, relA, relR, ex,                          
                        rank=100, PE="median"):                                     
    # grid points                                                                   
    f     = 1.                                                                      
    rs    = np.array([5., 10., 20.])                                                
    gamma = np.array([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])                         
                                                                                    
    cove_1 = []; cove_2 = []; cove_3 = []                                           
    for _rs in rs:                                                                  
        for _g in gamma:                                                            
            true = [f, _g, _rs]                                                     
            data = np.genfromtxt(filepath + "statistics_" + ex +                    
                  ("_N%i_relunc%.2f_relM%.2f_relA%.2f_relR%.2f_f%.1fgamma%.1frs%.1f"
                   %(nBDs, rel_unc, relM, relA, relR, f, _g, _rs)), unpack=True)                                 
            if PE=="median":                                                        
                low  = np.array((data[6], data[7], data[8]))                        
                high = np.array((data[9], data[10], data[11]))                      
            else:                                                                   
                sys.exit("Need to implement other point estimates")                 
            one = f > low[0]                                                        
            two = f < high[0]                                                       
            cove_1.append(len(np.where((one==True) & (two==True))[0]))              
            one = _g > low[1]                                                       
            two = _g < high[1]                                                      
            cove_2.append(len(np.where((one==True) & (two==True))[0]))              
            one = _rs > low[2]                                                      
            two = _rs < high[2]                                                     
            cove_3.append(len(np.where((one==True) & (two==True))[0]))              
    xi = np.array([2.5, 7.5, 15, 25])                                               
    yi = np.array([0., 0.25, 0.75, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55])             
    xi, yi = np.meshgrid(xi, yi, indexing="ij")                                     
                                                                                    
    zi_1   = np.array(cove_1).reshape(len(rs), len(gamma))                          
    zi_2   = np.array(cove_2).reshape(len(rs), len(gamma))                          
    zi_3   = np.array(cove_3).reshape(len(rs), len(gamma))                          
    # return                                                                    
    return xi, yi, zi_1, zi_2, zi_3   


def __grid_coverage_all__(filepath, nBDs, rel_unc, ex="ex1",                        
             ax=False, CR="symmetric",                                              
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,                 
             rank=100):                                                             
    """                                                                             
    Plot coverage grid in (rs, gamma)                                               
    """                                                                             
                                                                                    
    norm = colors.BoundaryNorm(boundaries=np.arange(0, 100, 5), ncolors=256)    
                                                                                    
    xi, yi, zi_1, zi_2, zi_3 = coverage_f_gamma_rs(filepath, nBDs, rel_unc,     
                                              ex, rank=rank, CR=CR)                 
    rs = [2.5, 5., 10., 20., 25.]                                                   
    g  = [0., 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]                                    
    xi = np.linspace(np.min(rs), np.min(rs), 5)
    yi = np.meshgrid(np.min(g), np.max(g), 10)                                                 
    cmap="RdYlGn"                                                                   
    if ax==False:                                                                   
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))                                
    if plot_f==True:                                                                
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap=cmap)                      
        zi_c = np.vstack((zi_1[0], zi_1, zi_1[-1]))                                 
        CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")           
        ax.clabel(CS, inline=True, fontsize=12, fmt="%i")                           
    elif plot_g==True:                                                              
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap=cmap)                      
        zi_c = np.vstack((zi_2[0], zi_2, zi_2[-1]))                                 
        CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")           
        ax.clabel(CS, inline=True, fontsize=12, fmt="%i")                           
    else:                                                                           
        im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap=cmap)                      
        zi_c = np.vstack((zi_3[0], zi_3, zi_3[-1]))                                 
        CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")           
        ax.clabel(CS, inline=True, fontsize=12, fmt="%i")                           
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
                                                                                    
    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_i$=%i"                          
                            %(int(np.log10(nBDs)), int(rel_unc*100))                
                            + "$\% $"),                                             
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))   
    plt.setp(text_box.patch, facecolor="white")                                     
    ax.add_artist(text_box)                                                         
                                                                                    
    ax.set_xticks([5., 10., 20.])                                                   
    ax.set_yticks([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])                            
                                                                                    
    for axis in ['top','bottom','left','right']:                                    
        ax.spines[axis].set_linewidth(2.5)                                          
                                                                                    
    # return                                                                        
    return im 

from scipy.interpolate import griddata

def grid_coverage_all(filepath, nBDs, rel_unc, ex="ex1",
             ax=False, CR="symmetric",
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,
             rank=100):
    """
    Plot coverage grid in (rs, gamma) 
    """

    #norm = colors.BoundaryNorm(boundaries=np.arange(0, 100, 5), ncolors=256)
    bounds = np.array([0, 10, 15, 20, 30, 40, 50, 68, 70, 75, 80, 85, 90, 95, 100])    
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

    xi, yi, zi_1, zi_2, zi_3 = coverage_f_gamma_rs(filepath, nBDs, rel_unc,
                                              ex, rank=rank, CR=CR)
    rs = [2.5, 5., 10., 20., 25.]
    g  = [0., 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    #xi_c, yi_c = np.meshgrid(rs, g)
    
    xi_c = np.linspace(np.min(rs), np.max(rs), 10)
    yi_c = np.linspace(np.min(g), np.max(g), 10)

    cmap="RdYlGn"
    if ax==False:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if plot_f==True:
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap=cmap)
        #zi_c = np.vstack((zi_1[0], zi_1, zi_1[-1]))
        #CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")
        x, y = np.meshgrid([5, 10, 20], [0, 0.5, 1., 1.1, 1.2, 1.3, 1.4, 1.5], indexing="ij")
        points = np.array((np.ravel(x), np.ravel(y))).T                         
        xi_c, yi_c = np.meshgrid(np.linspace(2.5, 25, 5), np.linspace(0., 1.5, 5), indexing="ij")
        values = np.array((np.ravel(xi_c), np.ravel(yi_c))).T                   
        zi_c = griddata(points, np.ravel(zi_3), values, method="nearest") 
        CS = ax.contour(xi_c, yi_c, zi_c.reshape(5, 5), levels=[68, 100], color="k") 
        ax.clabel(CS, inline=True, fontsize=10, fmt="%i")
    elif plot_g==True:
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap=cmap)
        #zi_c = np.vstack((zi_2[0], zi_2, zi_2[-1]))
        #CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")
        x, y = np.meshgrid([5., 10, 20.], [0, 0.5, 1., 1.1, 1.2, 1.3, 1.4, 1.5], indexing="ij")
        points = np.array((np.ravel(x), np.ravel(y))).T
        xi_c, yi_c = np.meshgrid(np.linspace(2.5, 25., 5), np.linspace(0., 1.5, 5), indexing="ij")
        values = np.array((np.ravel(xi_c), np.ravel(yi_c))).T
        zi_c = griddata(points, np.ravel(zi_2), values, method="nearest")
        CS = ax.contour(xi_c, yi_c, zi_c.reshape(5, 5), levels=[68, 100], color="k")          
        ax.clabel(CS, inline=True, fontsize=10, fmt="%i")
    else:
        im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap=cmap)
        #zi_c = np.vstack((zi_3[0], zi_3, zi_3[-1]))
        #CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68, 100], colors="k")
        x, y = np.meshgrid([5, 10, 20], [0, 0.5, 1., 1.1, 1.2, 1.3, 1.4, 1.5], indexing="ij")
        points = np.array((np.ravel(x), np.ravel(y))).T                         
        xi_c, yi_c = np.meshgrid(np.linspace(2.5, 25, 5), np.linspace(0., 1.5, 5), indexing="ij")
        values = np.array((np.ravel(xi_c), np.ravel(yi_c))).T                   
        zi_c = griddata(points, np.ravel(zi_3), values, method="nearest") 
        CS = ax.contour(xi_c, yi_c, zi_c.reshape(5, 5), levels=[68, 100], color="k")
        ax.clabel(CS, inline=True, fontsize=10, fmt="%i")
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

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_i$=%i"
                            %(int(np.log10(nBDs)), int(rel_unc*100))
                            + "$\% $"),
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18))
    plt.setp(text_box.patch, facecolor="white")
    ax.add_artist(text_box)

    ax.set_xticks([5., 10., 20.])
    ax.set_yticks([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)

    # return
    return im

def grid_coverage_each(filepath, nBDs, rel_unc, relM, relA, relR,
             ex="ex3",                  
             ax=False, PE="median",                                            
             plot_f=True, plot_g=False, ylabel=False, xlabel=False,            
             rank=100):                                                        
    """                                                                        
    Plot coverage grid in (rs, gamma)                                          
    """                                                                        
                                                                               
    norm = colors.BoundaryNorm(boundaries=np.arange(0, 100, 5), ncolors=256)   
                                                                                    
    xi, yi, zi_1, zi_2, zi_3 = coverage_f_gamma_rs_each(filepath, nBDs, 
                                            rel_unc, relM, relA, relR,
                                            ex, rank=rank, PE=PE)                 
    rs = [2.5, 5., 10., 20., 25.]                                                   
    g  = [0., 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]                                    
    xi_c, yi_c = np.meshgrid(rs, g)                                                 
    cmap="RdYlGn"                                                                   
    if ax==False:                                                                   
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))                                
    if plot_f==True:                                                                
        im = ax.pcolormesh(xi, yi, zi_1, norm=norm, cmap=cmap)                      
        zi_c = np.vstack((zi_1[0], zi_1, zi_1[-1]))                                 
        CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68], colors="k")                
        ax.clabel(CS, inline=True, fontsize=12, fmt="%i")                           
    elif plot_g==True:                                                              
        im = ax.pcolormesh(xi, yi, zi_2, norm=norm, cmap=cmap)                      
        zi_c = np.vstack((zi_2[0], zi_2, zi_2[-1]))                                 
        CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68], colors="k")                
        ax.clabel(CS, inline=True, fontsize=12, fmt="%i")                           
    else:                                                                           
        im = ax.pcolormesh(xi, yi, zi_3, norm=norm, cmap=cmap)                      
        zi_c = np.vstack((zi_3[0], zi_3, zi_3[-1]))                                 
        CS = ax.contour(xi_c, yi_c, zi_c.T, levels=[68], colors="k")              
        ax.clabel(CS, inline=True, fontsize=12, fmt="%i")                           
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
                                                                               
    if np.abs(relM+relA+relR) < 1e-3:
        variable = "T"
        rel      = rel_unc
    elif np.abs(relA+relR+rel_unc) < 1e-3:
        variable = "M"
        rel      = relM
    elif np.abs(relR+rel_unc+relM) < 1e-3:
        variable = "A"
        rel      = relA
    else:
        variable = "R"
        rel      = relR

    text_box = AnchoredText((r"$N=10^{%i}$, $\sigma_%s$=%i"                    
                            %(int(np.log10(nBDs)), variable, int(rel*100))     
                            + "$\% $"),                                        
                            frameon=True, loc=3, pad=0.2, prop=dict(size=18)) 
    plt.setp(text_box.patch, facecolor="white")                                 
    ax.add_artist(text_box)                                                     
                                                                                
    ax.set_xticks([5., 10., 20.])                                               
    ax.set_yticks([0., 0.5, 1, 1.1, 1.2, 1.3, 1.4, 1.5])                        
                                                                                
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


def plot_corner(samples, nBDs, relT, relM, f, gamma, rs, i=1, smooth=1.):

    fig, axes = corner(samples, levels=(1-np.exp(-0.5), 1-np.exp(-2)), plot_datapoints=False, 
                       plot_density=False, fill_contours=False, smooth=smooth, color="green",
                       range=[(0., 1.01), (-0.01, 2.5), (0., 40.)])
    # plot KDE smoothed version of distributions
    for axidx, samps in zip([0, 4, 8], samples.T):
        kde   = gaussian_kde(samps)
        xvals = fig.axes[axidx].get_xlim()
        xvals = np.linspace(xvals[0], xvals[1], 100)
        fig.axes[axidx].plot(xvals, kde(xvals)/np.max(kde(xvals)), color="green", lw=2.5)    
    
    axes[0, 0].axvline(1., color="r", ls="--", lw=2.5)
    axes[1, 1].axvline(gamma, color="r", ls="--", lw=2.5)
    axes[2, 2].axvline(rs, color="r", ls="--", lw=2.5)
    axes[1, 0].scatter(f, gamma, marker="x", color="red", s=60)
    axes[2, 0].scatter(f, rs, marker="x", color="red", s=60)
    axes[2, 1].scatter(gamma, rs, marker="x", color="red", s=60)
        
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[2, 0].set_xlabel(r"$f$")
    axes[2, 0].set_ylabel(r"$r_s$ [kpc]")
    axes[2, 1].set_xlabel(r"$\gamma$")
    axes[2, 2].set_xlabel(r"$r_s$ [kpc]")
    
    colors = ['green']
    lines = [Line2D([0], [0], color=c, linewidth=2.5, linestyle='-') for c in colors]
    labels = ['N %i, relT=%0.1f, relM=%.1f' %(nBDs, relT, relM)]
    axes[0, 2].legend(lines, labels, fontsize=16)
    
    fig.savefig(("../../Figs/corner_ex15_N%irelT%.2frelM%.2f_g%.1frs%.1f_%i.png" %(nBDs, relT, relM, gamma, rs, i+1)), 
                bbox_inches="tight")



def plot_corner_each(samples, ex, nBDs, relT, relM, relA, relR, f, gamma, rs, 
                     i=1, smooth=1.):           
                                                                                    
    fig, axes = corner(samples, levels=(1-np.exp(-0.5), 1-np.exp(-2)), plot_datapoints=False, 
                       plot_density=False, fill_contours=False, smooth=smooth, color="green",
                       range=[(0., 1.01), (-0.01, 2.5), (0., 40.)])                 
    # plot KDE smoothed version of distributions                                    
    for axidx, samps in zip([0, 4, 8], samples.T):                                  
        kde   = gaussian_kde(samps)                                                 
        xvals = fig.axes[axidx].get_xlim()                                          
        xvals = np.linspace(xvals[0], xvals[1], 100)                                
        fig.axes[axidx].plot(xvals, kde(xvals)/np.max(kde(xvals)), color="green", lw=2.5)    
                                                                                    
    axes[0, 0].axvline(1., color="r", ls="--", lw=2.5)                              
    axes[1, 1].axvline(gamma, color="r", ls="--", lw=2.5)                           
    axes[2, 2].axvline(rs, color="r", ls="--", lw=2.5)                              
    axes[1, 0].scatter(f, gamma, marker="x", color="red", s=60)                     
    axes[2, 0].scatter(f, rs, marker="x", color="red", s=60)                        
    axes[2, 1].scatter(gamma, rs, marker="x", color="red", s=60)                    
                                                                                    
    axes[1, 0].set_ylabel(r"$\gamma$")                                              
    axes[2, 0].set_xlabel(r"$f$")                                                   
    axes[2, 0].set_ylabel(r"$r_s$ [kpc]")                                           
    axes[2, 1].set_xlabel(r"$\gamma$")                                              
    axes[2, 2].set_xlabel(r"$r_s$ [kpc]")                                           
                                                                                    
    colors = ['green']                                                          
    lines = [Line2D([0], [0], color=c, linewidth=2.5, linestyle='-') for c in colors]
    labels = ['N %i, relT=%0.1f, relM=%.1f' %(nBDs, relT, relM)]                
    axes[0, 2].legend(lines, labels, fontsize=16)                               
                                                                                
    fig.savefig(("../../Figs/corner_" + ex + 
                 "_N%irelT%.2frelM%.2frelA%.2frelR%.2f_g%.1frs%.1f_%i.png" 
                 %(nBDs, relT, relM, relA, relR, gamma, rs, i+1)),
                bbox_inches="tight") 
