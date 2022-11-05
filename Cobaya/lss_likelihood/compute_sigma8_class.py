import numpy as np
import json

from taylor_approximation import taylor_approximate


class Compute_Sigma8():
    """
    Computes sigma8 given a Taylor series (stored in a JSON file).
    """
    def __init__(self,s8_filename):
        json_file = open(s8_filename, 'r')
        emu = json.load( json_file )
        json_file.close()
        #
        self.lnA0    = emu['lnA0']
        self.x0s     = emu['x0']
        self.derivs0 = [np.array(ll) for ll in emu['derivs0']]
        del emu
        #
    def compute_sigma8(self,OmM,h,lnA):
        # The order should be <= the order computed by the emulator generator.
        # Higher order is more accurate, but slower.
        s8_emu = taylor_approximate([OmM,h],self.x0s,self.derivs0,order=3)[0]
        s8_emu*= np.exp(0.5*(lnA-self.lnA0))
        return(s8_emu)
