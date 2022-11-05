import numpy as np

from classy import Class




def compute_sigma8(pars,lnA0=3.036394):
    """Uses CLASS to compute sigma8 for the specified cosmology."""
    #
    OmegaM,h = pars
    lnAs     = lnA0
    #
    omega_b  = 0.02237
    omega_nu = 0.0006442
    omega_c  = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2
    ns       = 0.9649
    nnu      = 1
    nur      = 2.0328
    #
    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 20.,
        'z_pk': '0.0,10',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        "n_s": ns,
        "omega_ncdm": omega_nu,
        "N_ncdm": nnu,
        "N_ur": nur,
        'tau_reio': 0.0544,
        'omega_b': omega_b,
        'omega_cdm': omega_c}
    #
    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    sigma8 = pkclass.sigma8()
    #
    return sigma8
