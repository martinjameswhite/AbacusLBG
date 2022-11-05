#!/usr/bin/env python3
#
# Generate a Gaussian/disconnected covariance matrix
# given a theoretical model.
#

import numpy as np
import sys


def discov(kk,pk0,pk2,Vobs=1e9):
    """The disconnected covariance matrix, gven binned P_0(k) and P_2(k), assumed to
    already contain stochastic terms.  The k vector is assumed linear in k."""
    # The disconnected piece of the covariance matrix for autospectra is:
    # Cov[Pell0,Pell2] = (2ell0+1)(2ell2+1)/(Vobs.k^2.dk) (2\pi^2)
    #                  x 2 int_0^1 dmu LegendreP[ell0,mu].LegendreP[ell2,mu] P^2
    Nk   = kk.size
    dk   = kk[1]-kk[0]
    mu   = np.linspace(0,1,100)
    L0   = np.ones_like(mu)
    L2   = 0.5*(3*mu**2-1)
    pkmu = np.outer(pk0,L0) + np.outer(pk2,L2)
    #
    cov  = np.zeros( (2*Nk,2*Nk) )
    cov[:Nk,:Nk] = np.diag((2*0+1)*(2*0+1)*np.trapz(L0*L0*pkmu**2,x=mu,axis=1))
    cov[:Nk,Nk:] = np.diag((2*0+1)*(2*1+1)*np.trapz(L0*L2*pkmu**2,x=mu,axis=1))
    cov[Nk:,:Nk] = np.diag((2*1+1)*(2*0+1)*np.trapz(L2*L0*pkmu**2,x=mu,axis=1))
    cov[Nk:,Nk:] = np.diag((2*1+1)*(2*1+1)*np.trapz(L2*L2*pkmu**2,x=mu,axis=1))
    cov         /= Vobs * np.append(kk,kk)**2 * dk
    cov         *= (2*np.pi)**2
    return(cov)
    #




if __name__=="__main__":
    if len(sys.argv)!=3:
        raise RuntimeError("Usage: "+sys.argv[0]+" <model-file-name> <volume>")
    #
    kk,pk0,pk2 = np.loadtxt(sys.argv[1],unpack=True)
    Vobs       = float(sys.argv[2])
    cov        = discov(kk,pk0,pk2,Vobs)
    #
    with open("discov.txt","w") as fout:
        fout.write("# Disconnected covariance matrix for stacked P_ell(k) data.\n")
        fout.write("# Read theory model from "+sys.argv[1]+"\n")
        fout.write("# Assuming Vobs={:e}\n".format(Vobs))
        fout.write("# Covariance is {:d}x{:d}\n".format(cov.shape[0],cov.shape[1]))
        for i in range(cov.shape[0]):
            outstr = ""
            for j in range(cov.shape[1]):
                outstr += " {:20.10e}".format(cov[i,j])
            fout.write(outstr+"\n")
    #
