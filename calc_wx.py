#!/usr/bin/env python3
#
# Compute the angular cross-correlation function of a
# catalog of "target" objects against another set of
# spectroscopic objects, assuming a random catalog only
# of the targets.
# Uses the Davis-Peebles estimator.
#
#
# I was having "issues" with the Corrfunc routines misbehaving
# when passed float32 rather than float so there is also a
# routine for a brute-force computation.  This is only intended
# for use with small catalogs (including randoms).
#
import numpy as np
import os
#
from astropy.table import Table
#
from Corrfunc.mocks import DDtheta_mocks as Pairs



def ang2vec(dat):
    """Returns an "nhat array" given RA, DEC (deg.) in dat."""
    tt,pp     = np.radians(90-dat['DEC']),np.radians(dat['RA'])
    sint      = np.sin(tt)
    nhat      = np.zeros( (tt.size,3) )
    nhat[:,0] = sint*np.cos(pp)
    nhat[:,1] = sint*np.sin(pp)
    nhat[:,2] = np.cos(tt)
    return(nhat)
    #



def spam_calc_wx(spec,targ,rand,bins=None,dchi=50,fixed_chi0=None):
    """Does a brute-force pair count."""
    # Get number of targets and randoms.
    Nt,Nr = len(targ['RA']),len(rand['RA'])
    # Get weights, assign uniform weights if 'WT' key is missing.
    if 'WT' in spec.keys():
        swt = spec['WT']
    else:
        swt = np.ones_like(spec['RA'])
    if 'WT' in targ.keys():
        twt = targ['WT']
    else:
        twt = np.ones_like(targ['RA'])
    if 'WT' in rand.keys():
        rwt = rand['WT']
    else:
        rwt = np.ones_like(rand['RA'])
    # Get unit vectors pointing to each object.
    nspec,ntarg,nrand = ang2vec(spec),ang2vec(targ),ang2vec(rand)
    # Bin edges are specified in Mpc/h, if nothing
    # is passed in, do log-spaced bins.
    if bins is None:
        Nbin = 8
        bins = np.geomspace(0.5,30.,Nbin+1)
    # Do the brute force pair-distance computation.
    # First do D_sD_t.
    wts  = np.outer(swt,twt)
    cost = np.dot(nspec,ntarg.T).clip(-1.,1.)
    rval = spec['CHI'][:,None]*np.sqrt(2*(1-cost))
    DD,_ = np.histogram(rval,weights=wts,bins=bins)
    # then do D_sR.
    wts  = np.outer(swt,rwt)
    cost = np.dot(nspec,nrand.T).clip(-1.,1.)
    rval = spec['CHI'][:,None]*np.sqrt(2*(1-cost))
    DR,_ = np.histogram(rval,weights=wts,bins=bins)
    DR  += 1e-20 # Avoid divide-by-zero.
    # Now compute the cross-spectrum using Davis-Peebles.
    wx = float(Nr)/float(Nt) * DD/DR - 1.0
    # Return the binning and w_theta(R).
    return( (bins,wx) )
    #





def calc_wx(spec,targ,rand,bins=None,dchi=50,fixed_chi0=None):
    """Does the work of calling CorrFunc."""
    # Get number of threads, datas and randoms.
    nthreads = int( os.getenv('OMP_NUM_THREADS','1') )
    Nt,Nr    = len(targ['RA']),len(rand['RA'])
    # Bin edges are specified in Mpc/h, if nothing
    # is passed in, do log-spaced bins.
    if bins is None:
        Nbin = 8
        bins = np.geomspace(0.5,30.,Nbin+1)
    # RA and DEC should be in degrees, and all arrays should
    # be the same type and it seems as if they need to be
    # 'float' and not e.g. 'float32'.  Ensure this now.
    sra = np.ascontiguousarray(spec['RA' ]).astype('float')
    sdc = np.ascontiguousarray(spec['DEC']).astype('float')
    tra = np.ascontiguousarray(targ['RA' ]).astype('float')
    tdc = np.ascontiguousarray(targ['DEC']).astype('float')
    rra = np.ascontiguousarray(rand['RA' ]).astype('float')
    rdc = np.ascontiguousarray(rand['DEC']).astype('float')
    # We'll need the distances to the spectroscopic sample.
    sch = spec['CHI']
    chimin,chimax = np.min(sch),np.max(sch)
    # Now do the pair counting in steps of chi.
    st   = np.zeros(len(bins)-1)
    sr   = np.zeros(len(bins)-1) + 1e-15
    chi0 = chimin
    while chi0<chimax:
        if fixed_chi0 is None:
            # For observational data want to use the proper
            # angle->distance conversion.
            tbins = bins/(chi0+0.5*dchi) * 180./np.pi # deg.
        else:
            # But for simpler simulations which have converted
            # using a fixed distance, make the same approximation.
            tbins = bins/fixed_chi0 * 180./np.pi # deg.
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
    if sr.any()<=0: raise RuntimeError("Have sr<=0: "+str(sr))
    wx = float(Nr)/float(Nt) * st/sr - 1.0
    # Return the binning and w_theta(R).
    return( (bins,wx) )
    #
