#!/usr/bin/env python3
#
# Compute the correlation function of a catalog of objects.
#
import numpy as np
import os
#
from astropy.table import Table
#
from Corrfunc.mocks import DDsmu_mocks as Pairs
from Corrfunc.utils import convert_3d_counts_to_cf



def calc_xi(dat,ran,bins=None):
    """Does the work of calling CorrFunc."""
    # Get number of threads, datas and randoms.
    nthreads = int( os.getenv('OMP_NUM_THREADS','1') )
    Nd,Nr    = len(dat['RA']),len(ran['RA'])
    # RA and DEC should be in degrees, and all arrays should
    # be the same type.  Ensure this now.
    pp  = 'pair_product'
    dra = dat['RA' ].astype('float32')
    ddc = dat['DEC'].astype('float32')
    dcz = dat['CHI'].astype('float32')
    if not 'WT' in dat.keys():
        dwt = np.ones_like(dat['RA'])
    else:
        dwt = dat['WT'].astype('float32')
    rra = ran['RA' ].astype('float32')
    rdc = ran['DEC'].astype('float32')
    rcz = ran['CHI'].astype('float32')
    if not 'WT' in ran.keys():
        rwt = np.ones_like(ran['RA'])
    else:
        rwt = ran['WT'].astype('float32')
    # Bin edges are specified in Mpc/h, if nothing
    # is passed in, do log-spaced bins.
    if bins is None:
        Nbin = 5
        bins = np.logspace(-0.5,1.5,Nbin+1)
    nmu_bins = 8
    # do the pair counting, then convert to xi(s,mu).
    # Cosmology=2 is Planck, but this isn't used.
    # specifying is_comoving_dist says "CZ" is really a comoving distance.
    DD = Pairs(1,2,nthreads,1.0,nmu_bins,bins,RA1=dra,DEC1=ddc,CZ1=dcz,\
               weights1=dwt,weight_type=pp,is_comoving_dist=True)
    RR = Pairs(1,2,nthreads,1.0,nmu_bins,bins,RA1=rra,DEC1=rdc,CZ1=rcz,\
               weights1=rwt,weight_type=pp,is_comoving_dist=True)
    DR = Pairs(0,2,nthreads,1.0,nmu_bins,bins,RA1=dra,DEC1=ddc,CZ1=dcz,\
               RA2=rra,DEC2=rdc,CZ2=rcz,\
               weights1=dwt,weights2=rwt,weight_type=pp,is_comoving_dist=True)
    xi = convert_3d_counts_to_cf(Nd,Nd,Nr,Nr,DD,DR,DR,RR)
    # Return the binning and xi(s,mu).
    return( (bins,xi) )
    #
