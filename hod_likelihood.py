#
# The likelihood module.
#

import numpy as np
import sys
import os

from hod_model import HODmodel

from scipy.interpolate import InterpolatedUnivariateSpline as Spline




class Likelihood():
    """A class for computing likelihoods."""
    def __init__(self,rank):
        """This sets up the likelihood code."""
        # Load the model class, and evaluate it at the
        # start point to force loading of the data.
        self.model = HODmodel("hod_big.yaml")
        hodpar,_   = self.startpar()
        tt = self.model(hodpar)
        # Load data files, covariance, etc.
        self.loadData()
        # Invert the covariance matrix.
        self.cinv= np.linalg.inv(self.cov)
        # Set the header.
        hdr  = "# HOD model using AbacusHOD.\n"
        hdr += self.model.paramNames()
        for k in ['SimName',\
                  'BoxSizeHMpc','ParticleMassHMsun',\
                  'Omega_M','H0','Redshift']:
            hdr += "# {:>20s} : ".format(k)+str(self.model.meta[k])+"\n"
        self.header = hdr
        # Print some useful information.
        if rank==0:
            print("Likelihood initialized.",flush=True)
        #
    def loadData(self):
        """Loads the data and covariance."""
        # First load the data.
        dd = np.loadtxt("hod_fit_dat.txt")
        # Generate the data vector as stacked multipoles.
        self.xx = dd[:,0]
        self.dd = dd[:,1]
        if False:
            # Useful for testing.
            p,_     = self.startpar()
            self.dd = self.observe(self.model(p))
        # Now load the covariance matrix.
        self.cov= np.loadtxt("hod_fit_cov.txt")
        #
    def startpar(self):
        """Returns a starting position and scatter for the parameters."""
        # Scatter is "fractional".
        pars = np.array([12.15,10.00,0.30,1.00,0.75])
        dpar = np.array([ 0.01, 0.02,0.02,0.02,0.02])
        return( (pars,dpar) )
        #
    def prior_chi2(self,p):
        """Returns priors, as a contribution to chi^2."""
        ret = 0.0
        return(ret)
        #
    def observe(self,tt):
        """Converts theory to binned observation."""
        ss   = Spline(tt[:,0],tt[:,1])
        rfac = np.sqrt(self.xx[1]/self.xx[0])
        obs  = np.zeros_like(self.xx)
        for i in range(self.xx.size):
            rmin    = self.xx[i] / rfac
            rmax    = self.xx[i] * rfac
            rval    = np.linspace(rmin,rmax,32)
            obs[i]  = np.trapz(rval*ss(rval),x=rval)
            obs[i] /= np.trapz(rval,x=rval)
        return(obs)
        #
    def __call__(self,p):
        """Returns the log-likelihood given some parameters
        (here a NumPy array).
        Eventually we're going to want to pass either a class or a dictionary
        of the parameters and then only extract the appropriate bits for each
        step in the likelihood evaluation."""
        # Check if we're out-of-bounds, in which case skip theory call.
        if self.model.paramsInvalid(p):
            return(np.array([-1e8-1e3*np.sum(p**2)]))
        else:
            tt = self.model(p)
        # Generate the "observed" theory model.
        thy  = self.observe(tt)
        # and compute chi^2.
        chi2 = np.dot(self.dd-thy,np.dot(self.cinv,self.dd-thy))
        chi2+= ( (self.model.nbar-7e-4)/1e-4 )**4
        chi2+= self.prior_chi2(p)
        return(-0.5*chi2)
    #
