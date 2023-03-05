#
# HOD model.
#

import numpy as np
import yaml
import json
import sys
import os

from abacusnbody.hod.abacus_hod import AbacusHOD
from abacusnbody.metadata       import get_meta

from scipy.interpolate import InterpolatedUnivariateSpline as Spline





class HODmodel():
    """The class that handles model predictions."""
    def paramNames(self):
        """Returns a string of the parameter names."""
        header = "# Pars: lgMcut,plateau,sigma,kappa,alpha\n"
        return(header)
    def paramsInvalid(self,p):
        """Returns true if the parameters lie outside of some "sanity check"
        ranges.  Triggers lnL to return a very low log-likelihood."""
        ret = False
        if (p[0]<11.)|(p[0]>14.): ret=True
        if (p[1]<0.5)|(p[1]>1e2): ret=True
        if (p[2]<0.0)|(p[2]>3.0): ret=True
        if (p[3]<0.1)|(p[3]>1e2): ret=True
        if (p[4]<0.3)|(p[4]>1.5): ret=True
        return(ret)
        #
    def make_ball(self,hodpars):
        """Generate the "ball" and run the HOD code."""
        lgMcut,plateau,sigma,kappa,alpha = hodpars
        self.HOD_params['LRG_params']['logM_cut'] = lgMcut
        self.HOD_params['LRG_params']['logM1'   ] = lgMcut+np.log10(plateau)
        self.HOD_params['LRG_params']['sigma'   ] = sigma
        self.HOD_params['LRG_params']['kappa'   ] = kappa
        self.HOD_params['LRG_params']['alpha'   ] = alpha
        if self.newBall is None:
            self.newBall  = AbacusHOD(self.sim_params,\
                                      self.HOD_params,\
                                      self.clustering_params)
        else:
            for k in ['logM_cut','logM1','sigma','kappa','alpha']:
                self.newBall.tracers['LRG'][k] = \
                    self.HOD_params['LRG_params'][k]
        want_rsd,write_to_disk = True,False
        want_rsd,write_to_disk = False,False
        self.mock_dict= self.newBall.run_hod(self.newBall.tracers,\
                                             want_rsd,write_to_disk,\
                                             Nthread=self.nthread)
        self.nobj  = self.mock_dict['LRG']['mass'].size
        self.nbar  = self.nobj/self.Lbox**3
        self.ncen  = self.mock_dict['LRG']['Ncent']
        self.fsat  = 1-float(self.ncen)/float(self.nobj)
        #
    def __call__(self,p):
        """The model prediction for parameter set p."""
        # Generate the "box of galaxies".
        self.make_ball(p)
        # If there are too many, downsample.
        maxobj  = 8000000 # Should be an integer.
        if self.nobj>maxobj:
            rng  = np.random.default_rng()
            inds = rng.choice(self.nobj,size=maxobj,replace=False)
            for k in ['x','y','z','vx','vy','vz','mass','id']:
                self.mock_dict['LRG'][k] = self.mock_dict['LRG'][k][inds]
        rpbins     = np.logspace(self.bin_params['logmin'],\
                                 self.bin_params['logmax'],\
                                 self.bin_params['nbins']+1)
        # Note pimax and pi_bin_size are ints.
        pimax,dpi  = self.clustering_params['pimax'],\
                     self.clustering_params['pi_bin_size']
        xiell = self.newBall.compute_multipole(self.mock_dict,\
                                               rpbins,pimax,dpi)['LRG_LRG']
        # work out the rp and pi bin centers (assume log binning as above)
        Rcen = np.sqrt(rpbins[1:]*rpbins[:-1])
        Zcen = np.arange(0.0,float(pimax),dpi) + 0.5*dpi
        wpR  = xiell[0*len(Rcen):1*len(Rcen)]
        xi0  = xiell[1*len(Rcen):2*len(Rcen)]
        xi2  = xiell[2*len(Rcen):3*len(Rcen)]
        # Let's save these values in a dictionary.
        self.d = {}
        self.d['par' ] = p.tolist()
        self.d['wpR' ] = wpR.tolist()
        self.d['xi0' ] = xi0.tolist()
        ##self.d['xi2' ] = xi2.tolist()
        self.Rcen      = Rcen.tolist()
        # Pack R,wp into a theory vector and return it.
        tt      = np.zeros( (Rcen.size,2) )
        tt[:,0] = Rcen.copy()
        ##tt[:,1] = wpR.copy()
        tt[:,1] = xi0.copy()
        return(tt)
        #
    def __init__(self,yaml_file):
        """This sets up the model."""
        # AbacusHOD is multi-threaded.  Save the number of threads.
        self.nthread = int(os.getenv("OMP_NUM_THREADS",1))
        # Load the config file and parse in relevant parameters
        config = yaml.safe_load(open(yaml_file))
        self.sim_params = config['sim_params']
        self.HOD_params = config['HOD_params']
        self.clustering_params = config['clustering_params']
        # Get the metaparameters for the simulation.
        self.meta = get_meta(self.sim_params['sim_name'],\
                             redshift=self.sim_params['z_mock'])
        self.Lbox = float(self.meta['BoxSizeHMpc'])
        # Load the rp pi binning from the config file.
        self.bin_params = self.clustering_params['bin_params']
        # Set "newBall" to be None so the code knows it needs
        # to generate it.
        self.newBall = None
        #
