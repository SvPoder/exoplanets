import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.misc import derivative
from astropy.constants import R_jup, M_jup, G, sigma_sb
from utils import gNFW_rho, vc

_sigma_sb = sigma_sb.value
_G        = G.value

def derivativeTDM_wrt_M(r, f=1, R=R_jup.value, M=M_jup.value, Rsun=8.178,
                        parameters=[1., 20., 0.42], v=None, epsilon=1):
    """
    Return (analytical) derivative of DM temperature wrt mass @ 
    (f, gamma, rs, rho0, r, M, R) [K/kg]
    """
    # escape velocity
    vesc   = np.sqrt(2*_G*M/R)*1e-3 # km/s
    if v:
        _vD = v
        #print(_vD, "here i am")
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, parameters) # km/s
        
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho(Rsun, r, parameters) # GeV/cm3

    conversion_into_w = 0.16021766 
    
    # DM temperature^-3 [1/K^3]
    T_DM3 = np.power((f*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                     conversion_into_w)/(4*_sigma_sb*epsilon), -3./4.)
    
    #print(T_DM3)
    
    conversion_into_K_vs_kg = 1.60217e-7
    # return 
    return (T_DM3*3./16.*np.sqrt(8./3./np.pi)*f/_sigma_sb/epsilon*_rhoDM*_G/_vD/R*
            conversion_into_K_vs_kg
           )


def derivativeTDM_wrt_r(r, f=1, R=R_jup.value, M=M_jup.value, Rsun=8.178,
                        parameters=[1., 20., 0.42], v=None, epsilon=1):
    """
    Return (analytical) derivative of DM temperature wrt r @ 
    (f, gamma, rs, rho0, r, M, R) [K/kpc]
    
    Assumption: DM velocity and velocity dispersion constant!
    """
    # escape velocity
    vesc   = np.sqrt(2*_G*M/R)*1e-3 # km/s
    if v:
        _vD = v
        #print(_vD, "here i am")
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, parameters) # km/s
        
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho(Rsun, r, parameters) # GeV/cm3

    conversion_into_w = 0.16021766 
    
    # DM temperature [K]
    T_DM = np.power((f*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
                     conversion_into_w)/(4*_sigma_sb*epsilon), 1./4.)
    
    return(0.25*T_DM*(-parameters[0]/r - (3-parameters[0])/(parameters[1] + r))
           )

def derivativeTint_wrt_A(M, A, points, values, size=7000, h=0.001):
    """
    Return (numerical) derivative of intrinsic temperature wrt Age [K/Gyr]
    
    Input
    -----
        M : mass [Msun]
        A : age [Gyr]
    """   
    ages   = np.linspace(1., 10., size)
    mass   = np.ones(size)*M
    xi     = np.transpose(np.asarray([ages, mass]))
    Teff   = griddata(points, values, xi)
    # return
    return derivative(interp1d(ages, Teff), A, dx=h)

def derivativeTint_wrt_M(M, A, points, values, size=7000, h=0.001):
    """
    Return (numerical) derivative of intrinsic temperature wrt Age [K/Gyr]
    
    Input
    -----
        M : mass [Msun]
        A : age [Gyr]
    """   
    ages   = np.ones(size)*A
    mass   = np.linspace(0.013, 0.053, size)
    xi     = np.transpose(np.asarray([ages, mass]))
    Teff   = griddata(points, values, xi)
    # return
    return derivative(interp1d(mass, Teff), M, dx=h)
