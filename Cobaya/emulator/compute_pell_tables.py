import numpy as np

from classy                         import Class
from linear_theory                  import f_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.61,0.01)) )

# Reference Cosmology:
z       = 3.00
Omega_M = 0.3151917236639384
h       = 0.6736
fb      = 0.02237/h**2/Omega_M
ns      = 0.9649
omeganu = 0.0006442
lnAs    = 3.036394
speed_of_light = 2.99792458e5




pkparams = {
    'output': 'mPk',
    'P_k_max_h/Mpc': 50.,
    'z_pk': '0.0,10',
    'A_s': np.exp(lnAs)*1e-10,
    'n_s': 0.9649,
    'h': h,
    "omega_ncdm": omeganu,
    "N_ncdm": 1.0,
    "N_ur": 2.0328,
    'tau_reio': 0.0544,
    'omega_b': h**2 * fb * Omega_M,
    'omega_cdm': h**2 * (1-fb) * Omega_M}

import time
t1 = time.time()
pkclass = Class()
pkclass.set(pkparams)
pkclass.compute()

Hz_fid   = pkclass.Hubble(z) * speed_of_light / h   # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
chiz_fid = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 

print("Hz_fid=",Hz_fid,", chiz_fid=",chiz_fid)

def compute_pell_tables(pars, z=3.00, fid_dists= (Hz_fid,chiz_fid), ap_off=False ):
    OmegaM, h, sigma8 = pars
    Hzfid, chizfid    = fid_dists
    #
    omega_b  = 0.02237
    omega_nu = 0.0006442
    omega_c =  OmegaM*h**2 - omega_b - omega_nu
    lnAs     = 3.036394
    ns       = 0.9649
    nnu      = 1
    nur      = 2.0328
    #
    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 50.,
        'z_pk': '0.0,10',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        'N_ur': nur,
        'N_ncdm': nnu,
        'omega_ncdm': omega_nu,
        'tau_reio': 0.0544,
        'omega_b': omega_b,
        'omega_cdm': omega_c}
    #
    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    #
    # Caluclate AP parameters
    Hz   = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    apar, aperp = Hzfid / Hz, chiz / chizfid
    
    if ap_off:
        apar, aperp = 1.0, 1.0
    
    # Calculate growth rate
    fnu = pkclass.Omega_nu / pkclass.Omega_m()
    f   = f_of_a(1/(1.+z), OmegaM=OmegaM) * (1 - 0.6 * fnu)

    # Calculate and renormalize power spectrum
    # We compute this at z=1 and shift it to z
    ss = pkclass.scale_independent_growth_factor(z)
    ss/= pkclass.scale_independent_growth_factor(1.0)
    ki = np.logspace(-3.0,1.0,200)
    pi = np.array( [pkclass.pk_cb(k*h,1.0)*h**3 for k in ki] )
    pi = (ss * sigma8/pkclass.sigma8())**2 * pi
    
    # Now do the RSD
    modPT = LPT_RSD(ki, pi, kIR=0.3,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
    modPT.make_pltable(f, kv=kvec, apar=apar, aperp=aperp, ngauss=3)
    
    return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable
