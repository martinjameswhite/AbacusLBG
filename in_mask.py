#
import numpy  as np
import healpy as hp
#
from astropy.table import Table
#



class SurveyMask:
    def __init__(self,maskfn):
        """Reads a FITS file containing a HealPix mask."""
        mskd       = Table.read(maskfn)
        self.nside = mskd.meta["HPXNSID"]
        self.nest  = mskd.meta["HPXNEST"]
        self.pixs  = mskd["HPXPIXEL"][~mskd["MASK"]]
    def __call__(self,ras,decs):
        """Returns a boolean array of whether the points pass the mask,
        with the points given by (RA,DEC) in degrees."""
        tt   = np.radians(90.-decs)
        pp   = np.radians(ras)
        pixs = hp.ang2pix(self.nside,tt,pp,nest=self.nest)
        return(np.in1d(pixs,self.pixs))
        #
