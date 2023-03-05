#!/usr/bin/env python3
#
# Runs a Monte-Carlo loop generating mock LBG catalogs and
# computing their clustering.
#
import numpy as np

from   fake_lbg  import MockLBG
from   calc_wx   import calc_wx
from   in_mask   import SurveyMask
from   rotate_to import rotate_to

from astropy.table     import Table
from scipy.interpolate import InterpolatedUnivariateSpline as Spline




if __name__=="__main__":
    # Set up a random number generator with a
    # fixed seed for reproducability.
    rng    = np.random.default_rng(1)
    # Set the name of the field we'll work with and load
    # the mask, random catalog and radial selection fn.
    fname= "xmmlss"
    sfn  = np.loadtxt("selection_fn.txt")
    mask = SurveyMask('clauds-{:s}-hpmask.fits'.format(fname))
    tt   = Table.read('clauds-{:s}-rands.fits'.format(fname))
    # Set the center of the field.
    cra = np.median(tt['RA'])
    cdc = np.median(tt['DEC'])
    print("Setting field center to ({:f},{:f})".format(cra,cdc),flush=True)
    # Mask the randoms.
    ww,ran     = mask(tt['RA'],tt['DEC']),{}
    ran['RA' ] = tt['RA' ][ww]
    ran['DEC'] = tt['DEC'][ww]
    # Define the mock catalog, shell and HOD.
    # 0  12.15  13.55   0.30   1.00   0.75   0.03   6.79e-04
    lbgs   = MockLBG('hod_big.yaml',None,4383.)
    params = {'logM_cut':12.15,'logM1':13.55,\
              'sigma':0.30,'kappa':1.00,'alpha':0.75}
    lbgs.set_hod(params)
    lbgs.generate()
    lbgs.assign_lum(0.25)
    # Select a field so we have access to the Lside etc.
    diam   = 4.5 * np.pi/180.
    lbgs.select(diam,[0.,0.,0.])
    chi0   = lbgs.d['chi0']
    Lside  = lbgs.d['Lside']
    Lx     = Lside/chi0 * 180./np.pi
    Ly     = Lside/chi0 * 180./np.pi
    ichi   = 1.0/  chi0 * 180./np.pi
    # Match the min/max values of chi.
    chimin,chimax = np.min(lbgs.zpos+chi0),np.max(lbgs.zpos+chi0)
    print("Raw 3D number density ",lbgs.d['nbar'])
    sampfn  = Spline(sfn[:,1],1e-3/lbgs.d['nbar']*sfn[:,2],k=1,ext='zeros')
    sampnrm = np.trapz(sampfn(sfn[:,1]),x=sfn[:,1])
    # Now do the MC loop.
    rval,wxs,ngals,fchis = None,[],[],[]
    for i in range(256):
        # Generate the galaxies.
        offset = rng.uniform(low=-0.5,high=0.5,size=3)
        lbgs.select(diam,offset)
        dat        = {}
        dat['RA' ] = lbgs.xpos*ichi
        dat['DEC'] = lbgs.ypos*ichi
        dat['CHI'] = lbgs.zpos+chi0
        # Apply radial selection function.
        rand = rng.uniform(low=0,high=1,size=dat['RA'].size)
        ww   = np.nonzero( rand<sampfn(dat['CHI']) )[0]
        dat['RA' ] = dat['RA' ][ww]
        dat['DEC'] = dat['DEC'][ww]
        dat['CHI'] = dat['CHI'][ww]
        # Rotate the objects to the field center and apply mask.
        nra,ndc    = rotate_to(dat['RA'],dat['DEC'],cra,cdc)
        ww         = mask(nra,ndc)
        dat['RA' ] = nra[ww]
        dat['DEC'] = ndc[ww]
        dat['CHI'] = dat['CHI'][ww]
        # Make a random selection of 10% of the objects as
        # the "spectroscopic targets" -- for now.
        rand = rng.uniform(low=0,high=1,size=dat['RA'].size)
        ww   = np.nonzero( rand<0.10 )[0]
        spec = {}
        spec['RA' ] = dat['RA' ][ww]
        spec['DEC'] = dat['DEC'][ww]
        spec['CHI'] = dat['CHI'][ww]
        avgfchi = np.mean( sampfn(spec['CHI'])/sampnrm )
        # compute the clustering.
        bins,wx = calc_wx(spec,dat,ran,dchi=1e5,fixed_chi0=chi0)
        rval    = np.sqrt( bins[:-1]*bins[1:] )
        wxs.append(wx)
        ngals.append(dat['RA'].size)
        fchis.append(avgfchi)
    wxs  = np.array(wxs)
    wavg = np.mean(wxs,axis=0)
    werr = np.std( wxs,axis=0)
    wcor = np.corrcoef(wxs,rowvar=False)
    navg = np.mean(np.array(ngals,dtype='float'))
    nerr = np.std( np.array(ngals,dtype='float'))
    favg = np.mean(np.array(fchis))
    ferr = np.std(np.array(fchis))
    # Now write out some results.
    with open("mc_lbg_wx.txt","w") as fout:
        fout.write("# Monte-Carlo calculation of wx using {:d} mocks.\n".\
                   format(wxs.shape[0]))
        fout.write("# Field "+fname+"\n")
        fout.write("# Centered on ({:.3f},{:.3f})\n".format(cra,cdc))
        fout.write("# Number density is {:.3e}\n".format(lbgs.d['nbar']))
        fout.write("# Have {:.1f}+/-{:.2f} LBGs/field.\n".\
                   format(navg,nerr))
        fout.write("# <fchi>={:e}+/-{:e}\n".format(favg,ferr))
        fout.write("# Correlation matrix is:\n")
        for i in range(rval.size):
            outstr = "#"
            for j in range(rval.size): outstr += " {:8.4f}".format(wcor[i,j])
            fout.write(outstr + "\n")
        fout.write("# {:>8s} {:>15s} {:>15s}\n".\
                   format("R[Mpc/h]","wx","dwx"))
        for i in range(rval.size):
            outstr = "{:10.3f} {:15.5e} {:15.5e}".format(rval[i],wavg[i],werr[i])
            fout.write(outstr+"\n")
    #
