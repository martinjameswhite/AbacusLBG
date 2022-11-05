import numpy as np
import sys

from classy import Class

def compute_fid_dists(z, OmegaM):
    # Reference Cosmology:
    h       = 0.6736
    fb      = 0.02237/h**2/OmegaM
    ns      = 0.9649
    omeganu = 0.0006442
    lnAs    = 3.036394
    speed_of_light = 2.99792458e5

    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 50.,
        'z_pk': '0.0,10',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        "omega_ncdm": omeganu,
        "N_ncdm": 1.0,
        "N_ur": 2.0328,
        'tau_reio': 0.0544,
        'omega_b': h**2 * fb * Omega_M,
        'omega_cdm': h**2 * (1-fb) * Omega_M}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()

    Hz_fid   = pkclass.Hubble(z) * speed_of_light   / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz_fid = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    
    return Hz_fid, chiz_fid
    #
