import sys
sys.path.append("/home/mariacst/exoplanets/.venv/lib/python3.6/site-packages")
sys.path.append("/home/mariacst/exoplanets/exoplanets/python/")
import imp
import mock_generation
imp.reload(mock_generation)
from mock_generation import mock_population_all, mock_population_all_Asimov
import numpy as np
from scipy.interpolate import griddata
from utils import T_DM, temperature_withDM
from astropy.constants import R_jup, M_sun
from scipy.stats import percentileofscore
from derivatives import derivativeTint_wrt_M, derivativeTint_wrt_A
from derivatives import derivativeTDM_wrt_M, derivativeTDM_wrt_r, derivativeTintana_wrt_A
from scipy.interpolate import interp1d

# Constant parameters & conversions ========================================== 
conv_Msun_to_kg = 1.98841e+30 # [kg/Msun]                              
# ============================================================================ 

def sigma_Tmodel2(r, M, A, sigma_r, sigma_M, sigma_A,                              
                  Tint, f, params, a, b, c,                        
                  v, R=R_jup.value, Rsun=8.178, epsilon=1):                        
    """                                                                         
    Return squared uncertainty in model temperature [UNITS??]                   
                                                                                
    Input:                                                                      
        r : Galactocentric distance [kpc]                                       
        M : mass [Msun]                                                         
        A : age [Gyr]                                                              
        a : interpolation function                                                 
        b : interpolation function                                                 
        c : = derivativeTintana_wrt_M - interpolation function [K/Msun]            
                                                                                
    Assumption: uncertainties in age, mass and galactocentric distance          
        are independent                                                         
    """                                                                            
    M_in_kg = M*conv_Msun_to_kg                                                    
                                                                                   
    _TDM = T_DM(r, M=M_in_kg, f=f, params=params, v=v)                             
    Ttot = np.power(_TDM**4 + Tint**4, 0.25)                                       
                                                                                   
    dervT_M = ((Tint/Ttot)**3* c(A) +                                              
               (_TDM/Ttot)**3*derivativeTDM_wrt_M(r, f, params, M, v))             
    
    # return                                                                    
    return (np.power((Tint/Ttot)**3*derivativeTintana_wrt_A(M, A, a, b)*sigma_A, 2)+ 
            np.power(dervT_M*sigma_M, 2)+                                          
            np.power((_TDM/Ttot)**3*derivativeTDM_wrt_r(r, f, params, M, v)*sigma_r, 2)) 


def lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, a, b, c,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    Tmodel = temperature_withDM(robs, Tint, M=Mobs*conv_Msun_to_kg, f=f,
                           p=[gamma, rs, rho0], v=v)
    
    _sigma_Tmodel2 = sigma_Tmodel2(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,       
                                   sigma_Aobs, Tint, f, [gamma, rs, rho0], a, b, c, v) 

    # return                                                                   
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b(Mobs, sigma_Mobs, Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, a, b, c):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    sigma_Tint2 = (np.power(derivativeTintana_wrt_A(Mobs, Aobs, a, b)*sigma_Aobs, 2) + 
                   np.power(c(Aobs)*sigma_Mobs, 2))
    
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigma_Tint2) + 
                       (Tint-Tobs)**2/(sigma_Tobs**2 + sigma_Tint2)) 


def TS(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, a, b, c,                         
       v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, a, b, c, 
                       v=v, R=R, Rsun=Rsun, rho0=rho0, 
                       epsilon=epsilon)
            +2*lnL_b(Mobs, sigma_Mobs, Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, 
                     a, b, c)
            )

def p_value_sb(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,    
               a, b, c, TS_obs, steps=300, Tmin=0., v=None, ex="fixedT10v100"):      
    """                                                                         
    Return p-value for gamma_k @ (f, rs) under s+b hypothesis                   
    """                                                                         
    # Compute TS pdf                                                            
    TS_k = np.zeros(steps) 
    for i in range(steps):                                                      
        # Generate experiments under s+b hypothesis  
        np.random.seed(i+1)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs, sigmaMobs, Aobs,               
        sigmaAobs) = mock_population_all(nBDs, relT, relM, relR, relA, 
                                           f, gamma_k, rs, Tmin=Tmin,v=v)     
        # Predicted intrinsic temperatures                                         
        xi       = np.transpose(np.asarray([Aobs, Mobs]))                       
        Teff     = griddata(points, values, xi)                                 
        # TS                                                                    
        TS_k[i] = TS(gamma_k, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,    
                     sigmaAobs, Tobs, sigmaTobs, Teff, a, b, c, v=v)  
    # return                                                                    
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 

def p_value_b(gamma_k, f, rs, nBDs, relT, relM, relR, relA, points, values,   
              a, b, c, TS_obs, steps=300, v=None, ex="fixedT10v100"):                      
    """                                                                         
    Return p-value for gamma_k @ (f, rs) under b hypothesis                     
    """                                                                         
    # Compute TS pdf                                                            
    TS_k = np.zeros(steps)
    for i in range(steps): 
        # Generate experiments under s+b hypothesis 
        np.random.seed(i+1)
        (robs, sigmarobs, Tobs, sigmaTobs, Mobs, sigmaMobs, Aobs,               
        sigmaAobs) = mock_population_all(nBDs, relT, relM, relR, relA, 
                                           0., 1., 1.)              
        # Predicted intrinsic temperatures                                      
        xi       = np.transpose(np.asarray([Aobs, Mobs]))                       
        Teff     = griddata(points, values, xi)                                 
        # TS                                                                    
        TS_k[i] = TS(gamma_k, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,    
                     sigmaAobs, Tobs, sigmaTobs, Teff, a, b, c, v=v)  
    # return
    return 100-percentileofscore(TS_k, TS_obs, kind="strict") 

def sigma_Tmodel2_obs(r, M, A, sigma_r, sigma_M, sigma_A, Tint, dervTint_M, 
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

def lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,
           Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,
           v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Return ln(L) assuming predicted temperature = DM + intrinsic
    """  
    # Calculate predicted temperature (DM + intrinsic)
    Tmodel = temperature_withDM(robs, Tint, M=Mobs*conv_Msun_to_kg, f=f,
                           p=[gamma, rs, rho0], v=v)
    
    _sigma_Tmodel2 = sigma_Tmodel2_obs(robs, Mobs, Aobs, sigma_robs, sigma_Mobs,   
                                   sigma_Aobs, Tint, dervTint_M, dervTint_A,   
                                   f, [gamma, rs, rho0], v=v, R=R, Rsun=Rsun,  
                                   epsilon=epsilon)                            
    # return                                                                   
    return -0.5*np.sum(np.log(sigma_Tobs**2 + _sigma_Tmodel2) + 
                       (Tmodel-Tobs)**2/(sigma_Tobs**2 + _sigma_Tmodel2)) 


def lnL_b_obs(sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
          dervTint_A):
    """
    Return ln(L) assuming predicted temperature = intrinsic
    """  
    
    sigma_Tint2 = (np.power(dervTint_A*sigma_Aobs, 2) + 
                   np.power(dervTint_M*sigma_Mobs, 2))
    
    # return
    return -0.5*np.sum(np.log(sigma_Tobs**2 + sigma_Tint2) + 
                       (Tint-Tobs)**2/(sigma_Tobs**2 + sigma_Tint2)) 

def _TS_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, Aobs, sigma_Aobs,  
       Tobs, sigma_Tobs, Tint, dervTint_M, dervTint_A,                         
       v=None, R=R_jup.value, Rsun=8.178, rho0=0.42, epsilon=1.):
    """
    Test statistics
    """
    # return
    return (-2.*lnL_sb_obs(gamma, f, rs, robs, sigma_robs, Mobs, sigma_Mobs, 
                       Aobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, dervTint_M, 
                       dervTint_A, v=v, R=R, Rsun=Rsun, rho0=rho0, 
                       epsilon=epsilon)
            +2*lnL_b_obs(sigma_Mobs, sigma_Aobs, Tobs, sigma_Tobs, Tint, 
                     dervTint_M, dervTint_A)
            )


def UL_at_rs(rs, f, nBDs, relT, relM, relR, relA, points, values,
             robs, sigmarobs, Mobs, sigmaMobs, Aobs, sigmaAobs, Tobs, 
             sigmaTobs, Teff, a, b, c, dervTint_A, dervTint_M,
             steps=300, rho0=0.42, v=None, gamma_min=0.01, gamma_max=2.95):
    # Grid in gamma
    gamma_k = np.linspace(gamma_min, gamma_max, 30) # change this?             
    for g in gamma_k:                                                          
        # Observed TS                                                          
        TS_obs = _TS_obs(g, f, rs, robs, sigmarobs, Mobs, sigmaMobs, Aobs,     
                         sigmaAobs, Tobs, sigmaTobs, Teff, dervTint_M, dervTint_A, 
                         v=v)                
        # s + b hypothesis                                                     
        _p_sb = p_value_sb(g, f, rs, nBDs, relT, relM, relR, relA,          
                           points, values, a, b, c, TS_obs, steps=steps, v=v)           
        #b hypothesis                                                          
        _p_b = p_value_b(g, f, rs, nBDs, relT, relM, relR, relA,            
                         points, values, a, b, c, TS_obs, steps=steps, v=v)             
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
    
    f         = 1.#float(sys.argv[1])
    nBDs      = 100
    sigma     = 0.1
    gamma_max = [1.5, 1.5, 1.5, 1.9]

    rs        = [5., 10, 20, 30]
    gamma_min = [0.01, 0.1, 0.2, 0.2]
    steps     = 200 # Need to vary
    
    relT = 0.1;
    ex   = "fixedT10v100"
    v    = 100. # km/s
    # Load ATMO2020 model                                                          
    path     = "/home/mariacst/exoplanets/exoplanets/data/"                          
    data     = np.genfromtxt(path + "./ATMO_CEQ_vega_MIRI.txt", unpack=True)         
    points   = np.transpose(data[0:2, :])                                            
    values   = data[2]
    gamma_up = np.ones(len(rs))*10.
    # Generate real observation
    seed      = 250
    np.random.seed(seed)
    (robs, sigmarobs, Tobs, sigmaTobs, Mobs,                                  
        sigmaMobs, Aobs, sigmaAobs) = mock_population_all_Asimov(nBDs, relT, 
                                      sigma, sigma, sigma, 0., 1., 1., 
                                      Tmin=0., v=v)                        
    xi   = np.transpose(np.asarray([Aobs, Mobs]))                               
    Teff = griddata(points, values, xi) 
    # Calculate derivatives Tint wrt Age and Mass                               
    dervTint_A = np.ones(nBDs)                                                  
    dervTint_M = np.ones(nBDs)                                                  
    size       = 7000                                                           
    h          = 0.001                                                          
    for i in range(nBDs):                                                       
        dervTint_A[i] = derivativeTint_wrt_A(Mobs[i], Aobs[i], points, values,  
                                         size=size, h=h)                        
        dervTint_M[i] = derivativeTint_wrt_M(Mobs[i], Aobs[i], points, values,  
                                         size=size, h=h)                                          

    # Load variables analytical derivatives Tint                                       
    path = "/home/mariacst/exoplanets/running/data/"
    masses, a, b = np.genfromtxt(path + "derv_ana_wrt_A.dat", unpack=True)             
    ages, c = np.genfromtxt(path + "derv_ana_wrt_M.dat", unpack=True)                  
    a_interp = interp1d(masses, a)                                                     
    b_interp = interp1d(masses, b)                                                     
    c_interp = interp1d(ages, c) 

    i = 0
    for _rs in rs:
        # UL
        gamma_up[i] = UL_at_rs(_rs, f, nBDs, relT, sigma, sigma, sigma, points, 
                               values, robs, sigmarobs, Mobs, sigmaMobs, Aobs,
                               sigmaAobs, Tobs, sigmaTobs, Teff, a_interp, b_interp, c_interp, 
                               dervTint_A, dervTint_M, steps=steps, v=v, 
                               gamma_min=gamma_min[i], gamma_max=gamma_max[i])
        print("%.1f  %.4f" %(_rs, gamma_up[i]))
        i+=1

    #print(gamma_up)
    output = np.array((np.array(rs), gamma_up))
    # save results
    np.savetxt("UL_" + ex + "_nBDs%i_f%.1f_steps%i_sigma%.1f_rs%.1fAsimov.dat" 
               %(nBDs, f, steps, sigma, _rs), 
               output.T, fmt="%.4f  %.4f")
