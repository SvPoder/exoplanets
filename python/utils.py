import numpy as np
from astropy.constants import R_jup, M_jup, G, sigma_sb
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
import astropy.units as u


def vc(Rsun, Rint, parameters):
    data = np.genfromtxt("../data/rc_e2bulge_R08.178_J_corr.dat", unpack=True)
    r = data[0]
    vB = data[1]
    data = np.genfromtxt("../data/rc_hgdisc_R08.178_corr.dat", unpack=True)
    vD = data[1]
    vDM = vgNFW(Rsun, r, parameters)
    vtot = np.sqrt(np.power(vB, 2) + np.power(vD, 2)+ np.power(vDM, 2))
    vtot_intp = interp1d(r, vtot)
    return vtot_intp(Rint)

def vgNFW(Rsun, R, parameters):
    """
    Rotation velocity for gNFW dark matter density profile
    """
    # gNFW parameters
    gamma = parameters[0]
    Rs    = parameters[1]
    rho0  = parameters[2] 
    v     = []; 
    for Rint in R:
        hyp=np.float(hyp2f1(3-gamma,3-gamma,4-gamma,-Rint/Rs))
        Integral=(-2**(2+3*gamma)*np.pi*Rint**(3-gamma)*(1+
                  Rsun*(1./Rs))**(3-gamma)*rho0*hyp)/(-3+gamma)
        v.append(np.sqrt(1.18997*10.**(-31.)*Integral/Rint)*3.08567758*10.**(16.))
    v = np.array(v,dtype=np.float64)      
    # Return
    return v

def gNFW_rho(Rsun, R, parameters):
    """
    Return gNFW density profile at r distance from the GC
    Denstiy has same units as local DM density rho0
    """
    # gNFW parameters
    gamma = parameters[0] 
    Rs    = parameters[1]
    rho0  = parameters[2]
    # Density profile
    rho   = rho0*np.power(Rsun/R, gamma)*np.power((Rs+Rsun)/(Rs+R), 3-gamma)    
    # Return
    return rho

def heat_DM(r, f=1, R=R_jup.value, M=M_jup.value, Rsun=8.178, 
            parameters=[1., 20., 0.42], v=None):
    """
    Heat flow due to DM capture and annihilation
    """
    vesc   = (np.sqrt(2*G*M/R)).value*1e-3 # m/s 
    if v:
        _vD = v
        #print(_vD, "here i am")
    else:
        _vD    = np.sqrt(3/2.)*vc(Rsun, r, parameters) # km/s
        #print("rC")
    _vDM   =  np.sqrt(8./(3*np.pi))*_vD # km/s
    _rhoDM = gNFW_rho(Rsun, r, parameters) # GeV/cm3

    conversion_into_w = 0.16021766 

    # return
    return (f*np.pi*R**2*_rhoDM*_vDM*(1+3./2.*np.power(vesc/_vD, 2))*
            conversion_into_w) # W

def temperature_withDM(r, heat_int, f=1, R=R_jup.value, M=M_jup.value, 
                parameters=[1., 20., 0.42], v=None, epsilon=1):
    """
    Exoplanet temperature : internal heating + DM heating
    """
    return (np.power((heat_int + heat_DM(r, f=f, R=R, M=M, 
                     parameters=parameters, v=v))/
                     (4*np.pi*R**2*sigma_sb.value*epsilon), 1./4.))

def temperature(heat, R, epsilon=1):
    return np.power(heat/(4*np.pi*R**2*sigma_sb*epsilon), 0.25)

def heat(temp, R, epsilon=1):
        return (4*np.pi*R**2*sigma_sb.value*temp**4*epsilon)

