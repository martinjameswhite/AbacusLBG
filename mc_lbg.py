#!/usr/bin/env python3
#
# Runs a Monte-Carlo loop generating mock LBG catalogs and
# computing their clustering.
#
import numpy as np
from   fake_lbg import MockLBG
from   calc_xi  import calc_xi





if __name__=="__main__":
    # Set up a random number generator with a
    # fixed seed for reproducability.
    rng    = np.random.default_rng(1)
    # Define the mock catalog, shell and HOD.
    # 0  12.15  13.55   0.30   1.00   0.75   0.03   6.79e-04
    lbgs   = MockLBG('hod_big.yaml',None,4383.)
    params = {'logM_cut':12.15,'logM1':13.55,\
              'sigma':0.30,'kappa':1.00,'alpha':0.75}
    lbgs.set_hod(params)
    lbgs.generate()
    lbgs.assign_lum(0.25)
    # Select a field so we have access to the Lside etc.
    diam   = 3.2 * np.pi/180.
    lbgs.select(diam,[0.,0.,0.])
    chi0   = lbgs.d['chi0']
    Lside  = lbgs.d['Lside']
    Lx     = Lside/chi0 * 180./np.pi
    Ly     = Lside/chi0 * 180./np.pi
    ichi   = 1.0/  chi0 * 180./np.pi
    # Now we want to determine the sampling fraction to
    # get the right angular number density.
    ntarget,nbar = 1000.0,[]
    for i in range(25):
        # Generate the galaxies.
        offset = rng.uniform(low=-0.5,high=0.5,size=3)
        lbgs.select(diam,offset)
        nobj   = lbgs.d['nkeep']
        nbar.append(nobj / (Lx*Ly))
    fsamp = ntarget / np.median(nbar)
    # Match the min/max values of chi.
    chimin,chimax = np.min(lbgs.zpos+chi0),np.max(lbgs.zpos+chi0)
    # Generate a uniform random catalog.
    # We offset the RA to eliminate negative RAs
    # just to avoid a warning.
    nran = 8000000
    ran  = {}
    ran['RA' ] = rng.uniform(low=-Lx/2.,high=Lx/2. ,size=nran) + Lx
    ran['DEC'] = rng.uniform(low=-Ly/2.,high=Ly/2. ,size=nran)
    ran['CHI'] = rng.uniform(low=chimin,high=chimax,size=nran)
    # apply mask.
    rad2 = ( (ran['RA']-Lx)**2 + (ran['DEC'])**2 )*(np.pi/180.)**2
    ran['RA' ] = ran['RA' ][rad2<diam**2/4]
    ran['DEC'] = ran['DEC'][rad2<diam**2/4]
    ran['CHI'] = ran['CHI'][rad2<diam**2/4]
    # Now do the MC loop.
    rval,xis,ngals = None,[],[]
    for i in range(256):
        # Generate the galaxies.
        offset = rng.uniform(low=-0.5,high=0.5,size=3)
        lbgs.select(diam,offset)
        dat        = {}
        dat['RA' ] = lbgs.xpos*ichi + Lx
        dat['DEC'] = lbgs.ypos*ichi
        dat['CHI'] = lbgs.zpos + chi0
        # downsample
        rand = rng.uniform(low=0,high=1,size=dat['RA'].size)
        ww   = np.nonzero( rand<fsamp )[0]
        dat['RA' ] = dat['RA' ][ww]
        dat['DEC'] = dat['DEC'][ww]
        dat['CHI'] = dat['CHI'][ww]
        # apply mask.
        rad2 = ( (dat['RA']-Lx)**2 + (dat['DEC'])**2 )*(np.pi/180.)**2
        dat['RA' ] = dat['RA' ][rad2<diam**2/4]
        dat['DEC'] = dat['DEC'][rad2<diam**2/4]
        dat['CHI'] = dat['CHI'][rad2<diam**2/4]
        # compute the clustering.
        bins,xi = calc_xi(dat,ran)
        rval    = np.sqrt( bins[:-1]*bins[1:] )
        xis.append(xi)
        ngals.append(dat['RA'].size)
    xis  = np.array(xis)
    xavg = np.mean(xis,axis=0)
    xerr = np.std( xis,axis=0)
    xcor = np.corrcoef(xis,rowvar=False)
    navg = np.mean(np.array(ngals,dtype='float'))
    nerr = np.std( np.array(ngals,dtype='float'))
    # Now write out some results.
    diam *= 180./np.pi
    area  = np.pi * (diam/2)**2
    with open("mc_lbg_xi.txt","w") as fout:
        fout.write("# Monte-Carlo calculation of xi using {:d} mocks.\n".\
                   format(xis.shape[0]))
        fout.write("# Field diameter is {:.2f}deg, area {:.2f}deg2.\n".\
                   format(diam,area))
        fout.write("# Sampling by {:.3f} to get {:.1f} LBGs/deg2\n".\
                   format(fsamp,ntarget))
        fout.write("# Have {:.1f}+/-{:.2f} LBGs/field.\n".\
                   format(navg,nerr))
        fout.write("# Correlation matrix is:\n")
        for i in range(rval.size):
            outstr = "#"
            for j in range(rval.size): outstr += " {:8.4f}".format(xcor[i,j])
            fout.write(outstr + "\n")
        fout.write("# {:>8s} {:>15s} {:>15s}\n".\
                   format("R[Mpc/h]","xi0","dxi0"))
        for i in range(rval.size):
            outstr = "{:10.3f} {:15.5e} {:15.5e}".format(rval[i],xavg[i],xerr[i])
            fout.write(outstr+"\n")
    #
