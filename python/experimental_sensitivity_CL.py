import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
sys.path.append("/home/mariacst/exoplanets/exoplanets/python/")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_check
import numpy as np
from scipy.interpolate import griddata
from utils import T_DM, temperature_withDM
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore
from derivatives import derivativeTint_wrt_M, derivativeTint_wrt_A
from derivatives import derivativeTDM_wrt_M, derivativeTDM_wrt_r

# Constant parameters & conversions ========================================== 
conv_Msun_to_kg = 1.98841e+30 # [kg/Msun]                              
# ============================================================================ 

def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A, Tint, dervTint_M, 
                  dervTint_A, f, params, v=None, R=R_jup.value, Rsun=8.178, 
                  epsilon=1):               
    """                                                                        
    Return squared uncertainty in model temperature [UNITS??]                  
                                                                               
    Input:                                                                     
        r : Galactocentric distance [kpc]                                      
        M : mass [Msun]                                                        
        A : age [Gyr]                                                          
                                                                               
    Assumption: uncertainties in age, mass and galactocentric distance         
        are independent                                                        
    """                                                                        
    M_in_kg = M*conv_Msun_to_kg                                                
                                                                               
    _TDM = T_DM(r, R=R, M=M_in_kg, Rsun=Rsun, f=f, params=params, v=v,         
                epsilon=epsilon)                                               
    Ttot = np.power(_TDM**4 + Tint**4, 0.25)                                   
                                                                               
    dervT_M = ((Tint/Ttot)**3*dervTint_M +                                     
               (_TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v=v, R=R,   
                                                  Rsun=Rsun,epsilon=epsilon))  
    # return                                                                   
    return (np.power((Tint/Ttot)**3*dervTint_A*sigma_A, 2)+                    
            np.power(dervT_M*sigma_M, 2)+                                      
            np.power((_TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v=v,  
                                  R=R, Rsun=Rsun, epsilon=epsilon)*sigma_r, 2))


def lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    Tmodel = temperature_withDM(robs, Tint, M=Mobs*conv_Msun_to_kg, f=f,
                           p=[gamma, rs, rho0], v=v)
    
    _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,   
                                   sigma_Aobs, Tint, dervTint_M, dervTint_A,   
                                   f, [gamma, rs, rho0], v=v, R=R, Rsun=Rsun,  
                                   epsilon=epsilon)                            
    # return                                                                   
    return -0.5*np.sum((Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b(sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
          dervTint_A):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    sigma_Tint2 = (np.power(dervTint_A*sigma_Aobs, 2) + 
                   np.power(dervTint_M*sigma_Mobs, 2))
    
    # return
    return -0.5*np.sum((Tint-Tobs)**2/(sigma_Tobs**2 + sigma_Tint2)) 


def TS(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,                         
       v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
                       dervTint_A, v=v, R=R, Rsun=Rsun, rho0=rho0, 
                       epsilon=epsilon)
            +2*lnL_b(sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, 
                     dervTint_M, dervTint_A)
            )

def p_value_sb(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,    
                   TS_obs, steps=300, Tmin=0., v=None):                             
    """                                                                         
    Return p-value for gamma_k @ (f, rs) under s+b hypothesis                   
    """                                                                         
    # Compute TS pdf                                                            
    TS_k = np.zeros(steps) 
    path = "/home/mariacst/exoplanets/exoplanets/python/data_der/"
    ex   = "fixedT100K"
    for i in range(steps):                                                      
        sigmaTobs = 100. # K
        # Generate experiments under s+b hypothesis  
        np.random.seed(i)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs, sigmaMobs, Aobs,               
        sigmaAobs) = mock_population_check(nBDs, sigmaTobs, relM, relR, relA, 
                                           f, gamma_k, rs, Tmin=Tmin,v=v)     
        # Predicted intrinsic temperatures                                         
        xi       = np.transpose(np.asarray([Aobs, Mobs]))                       
        Teff     = griddata(points, values, xi)                                 
        # Load derivatives Tint wrt Age and Mass  
        data = np.genfromtxt(path + "derivativeTint_" + ex + "_N%i_sigma%.1f_v%i.dat" 
                     %(nBDs, relM, i), unpack=True) 
        dervTint_A = data[0]                                              
        dervTint_M = data[1]                                              
        # TS                                                                    
        TS_k[i] = TS(gamma_k, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,    
                     sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M, dervTint_A,  
                     v=v)  
    # return                                                                    
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 

def p_value_b(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,   
              TS_obs, steps=300, v=None):                                     
    """                                                                         
    Return p-value for gamma_k @ (f, rs) under b hypothesis                     
    """                                                                         
    # Compute TS pdf                                                            
    TS_k = np.zeros(steps)
    path = "/home/mariacst/exoplanets/exoplanets/python/data_der/" 
    ex   = "fixedT100K"
    for i in range(steps): 
        sigmaTobs = 100. # K 
        # Generate experiments under s+b hypothesis 
        np.random.seed(i)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs, sigmaMobs, Aobs,               
        sigmaAobs) = mock_population_check(nBDs, sigmaTobs, relM, relR, relA, 
                                           0., 1., 1.)              
        # Predicted intrinsic temperatures                                      
        xi       = np.transpose(np.asarray([Aobs, Mobs]))                       
        Teff     = griddata(points, values, xi)                                 
        # Load derivatives Tint wrt Age and Mass                           
        data = np.genfromtxt(path + "derivativeTint_" + ex + "_N%i_sigma%.1f_v%i.dat" 
                     %(nBDs, relM, i), unpack=True)  
        dervTint_A = data[0]                        
        dervTint_M = data[1]                                              
        # TS                                                                    
        TS_k[i] = TS(gamma_k, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,    
                     sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M, dervTint_A, v=v)  
    # return
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 


def UL_at_rs(rs, f, nBDs, relT, relM, relR, relA, points, values,
             robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs, 
             sigmaTobs, Teff, dervTint_M, dervTint_A, 
             steps=300, rho0=0.42, v=None, gamma_min=0.01, gamma_max=2.95):
    # Grid in gamma
    gamma_k = np.linspace(gamma_min, gamma_max, 30) # change this?             
    for g in gamma_k:                                                          
        # Observed TS                                                          
        TS_obs = TS(g, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,     
                    sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M,              
                    dervTint_A, v=v)                                
        # s + b hypothesis                                                     
        _p_sb = p_value_sb(g, f, rs, nBDs, relT, relM, relR, relA,          
                           points, values, TS_obs, steps=steps, v=v)           
        #b hypothesis                                                          
        _p_b = p_value_b(g, f, rs, nBDs, relT, relM, relR, relA,            
                         points, values, TS_obs, steps=steps, v=v)             
        try:                                                                   
            CL = _p_sb / _p_b                                                  
        except ZeroDivisionError:                                              
            CL = 200.                                                          
        if CL < 0.05:                                                          
            gamma_up = g                                                    
            break   
    #return
    return gamma_up

if __name__=="__main__":
    f         = 1.
    rs        = float(sys.argv[1])
    gamma_min = float(sys.argv[2])
    ite       = 50 # Need to vary
    steps     = 200 # Need to vary

    nBDs=100; relT=0.10; relM=0.10; relR=0.10; relA=0.10
    ex = "fixedT100K"
    # Load ATMO2020 model                                                          
    path     = "/home/mariacst/exoplanets/exoplanets/data/"                          
    data     = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)         
    points   = np.transpose(data[0:2, :])                                            
    values   = data[2]
    gamma_up = np.ones(ite)*10.
    path = "/home/mariacst/exoplanets/exoplanets/python/data_der/"
    for i in range(ite):
        # Generate real observation
        sigmaTobs = 100. # K
        seed      = i + 350                                                        
        np.random.seed(seed)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs,                                       
        sigmaMobs, Aobs, sigmaAobs) = mock_population_check(nBDs, sigmaTobs, relM, 
                                      relR, relA, 0., 1., 1., Tmin=0., v=100.)        
        xi   = np.transpose(np.asarray([Aobs, Mobs]))                                  
        Teff = griddata(points, values, xi)                                            
        # Load derivatives Tint wrt Age and Mass  
        data = np.genfromtxt(path + "derivativeTint_" + ex + "_N%i_sigma%.1f_v%i.dat" 
                     %(nBDs, relM, seed), unpack=True) 
        dervTint_A = data[0]                                              
        dervTint_M = data[1]  
        # UL
        gamma_up[i] = UL_at_rs(rs, f, nBDs, relT, relM, relR, relA, points, 
                               values, robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                               sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M,
                               dervTint_A, steps=steps, v=100., 
                               gamma_min=gamma_min, gamma_max=2.7)
        print("%i  %.4f" %(i, gamma_up[i]))

    #print(gamma_up)
    # save results
    np.savetxt("UL_f%.1f_rs%.1f_steps%i.dat" %(f, rs, steps), gamma_up, 
               fmt="%.4f")
