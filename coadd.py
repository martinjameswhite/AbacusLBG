#!/usr/bin/env python3
#
# Coadd the power spectrum data across runs to produce a
# low(er) noise measurement.
#

import numpy as np
import json



def coadd_pk(icos=0,ihod=0,Nsim=5):
    """Does the work of averaging the P_ell(k).  Eventually this could be a
    robust average, but for now just average."""
    isim = 0
    fn   = "lbg_clustering_c{:03d}_ph{:03d}_s.json".format(icos,isim)
    lbg = json.load(open(fn,"r"))
    kk,pk0,pk2 = np.array(lbg['k']),\
                 np.array(lbg['mocks'][ihod]['pk0']),\
                 np.array(lbg['mocks'][ihod]['pk2'])
    for isim in range(1,Nsim):
        fn   = "lbg_clustering_c{:03d}_ph{:03d}_s.json".format(icos,isim)
        lbg = json.load(open(fn,"r"))
        pk0+= np.array(lbg['mocks'][ihod]['pk0'])
        pk2+= np.array(lbg['mocks'][ihod]['pk2'])
    pk0 /= Nsim
    pk2 /= Nsim
    return( (kk,pk0,pk2) )
    #







if __name__=="__main__":
    #
    icos = 0
    ihod = 0
    Nsim = 16
    kk,pk0,pk2 = coadd_pk(icos,ihod,Nsim)
    #
    with open("lbg_c{:03d}_h{:03d}_pkl.txt".format(icos,ihod),"w") as fout:
        fout.write("# Coadded P_ell(k) data.\n")
        fout.write("# Using c{:03d} and HOD {:d}.\n".format(icos,ihod))
        fout.write("# Averaging {:d} simulations.\n".format(Nsim))
        fout.write("# {:>13s} {:>15s} {:>15s}\n".format("k[h/Mpc]","P_0(k)","P_2(k)"))
        for i in range(kk.size):
            fout.write("{:15.5e} {:15.5e} {:15.5e}\n".format(kk[i],pk0[i],pk2[i]))
