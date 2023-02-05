#!/usr/bin/env python3
#
# Compute the angular cross-correlation function of a
# catalog of "target" objects against another set of
# spectroscopic objects, assuming a random catalog only
# of the targets.
# Uses the Davis-Peebles estimator.
#
import numpy as np
import os
#
from astropy.table import Table
#
from Corrfunc.mocks import DDtheta_mocks as Pairs
from Corrfunc.utils import convert_3d_counts_to_cf



def calc_wx(spec,targ,rand,bins=None,dchi=50):
    """Does the work of calling CorrFunc."""
    # Get number of threads, datas and randoms.
    nthreads = int( os.getenv('OMP_NUM_THREADS','1') )
    Nt,Nr    = len(targ['RA']),len(rand['RA'])
    # Bin edges are specified in Mpc/h, if nothing
    # is passed in, do log-spaced bins.
    if bins is None:
        Nbin = 5
        bins = np.logspace(0.0,1.5,Nbin+1)
    # RA and DEC should be in degrees, and all arrays should
    # be the same type.  Ensure this now.
    sra = spec['RA' ].astype('float32')
    sdc = spec['DEC'].astype('float32')
    tra = spec['RA' ].astype('float32')
    tdc = spec['DEC'].astype('float32')
    rra = rand['RA' ].astype('float32')
    rdc = rand['DEC'].astype('float32')
    # We'll need the distances to the spectroscopic sample.
    sch = spec['CHI']
    chimin,chimax = np.min(sch),np.max(sch)
    # Now do the pair counting in steps of chi.
    wx   = np.zeros(len(bins)-1)
    st   = np.zeros(len(bins)-1)
    sr   = np.zeros(len(bins)-1) + 1e-15
    chi0 = chimin
    while chi0<chimax:
        tbins = bins/(chi0+0.5*dchi) * 180./np.pi # deg.
        ww    = np.nonzero( (sch>=chi0)&(sch<chi0+dchi) )[0]
        if len(ww)>0:
            # do the pair counting.
            DD = Pairs(0,nthreads,tbins,\
                       RA1=sra[ww],DEC1=sdc[ww],RA2=tra,DEC2=tdc)
            DR = Pairs(0,nthreads,tbins,\
                       RA1=sra[ww],DEC1=sdc[ww],RA2=rra,DEC2=rdc)
            st += DD['npairs'].astype('float')
            sr += DR['npairs'].astype('float')
        chi0 += dchi
    wx = float(Nr)/float(Nt) * st/sr - 1.0
    # Return the binning and w_theta(R).
    return( (bins,wx) )
    #