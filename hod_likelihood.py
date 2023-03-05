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
            print("Initial theory model:",flush=True)
            for i in range(tt.shape[0]):
                print("{:12.4e} {:12.4e}".format(tt[i,0],tt[i,1]))
            obs = self.observe_wt(tt)
            print("Would be observed as:",flush=True)
            for i in range(self.xx.size):
                print("{:12.4e} {:12.4e} ({:12.4e})".\
                      format(self.xx[i],obs[i],self.dd[i]))
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
        # The real-space matter correlation function.
        self.xim= np.loadtxt("hod_fit_xim.txt")
        self.xmx= self.xim[-1,0]
        self.xim= Spline(self.xim[:,0],self.xim[:,1],ext='zeros')
        # The selection function: z,chi,p(chi)
        pchi      = np.loadtxt("hod_fit_pch.txt")
        pchi[:,2]/= np.trapz(pchi[:,2],x=pchi[:,1])
        self.pchi = Spline(pchi[:,1],pchi[:,2],ext='zeros')
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
    def extend_xir(self,tt,Nuse=-4):
        """Uses an analytic correlation function to extend the range of xi(r).
        This assumes constant bias for simplicity.  Returns a spline."""
        xv = self.xim(tt[Nuse:,0])
        B2 = np.mean(tt[Nuse:,1]/xv)
        rr = np.linspace(1.5*tt[-1,0],self.xmx,100)
        xv = np.append(tt[:,0],rr)
        yv = np.append(tt[:,1],B2*self.xim(rr))
        ss = Spline(xv,yv,ext='zeros')
        return(ss)
        #
    def wtheta(self,theta,xir,Npnt=750):
        """Returns w(theta) evaluated at (scalar!) theta.
        Uses Eq.(23) of https://arxiv.org/abs/astro-ph/0609165 ."""
        cost,dsum = np.cos(theta),0.0
        cmin,cmax = self.pchi.get_knots()[0],self.pchi.get_knots()[-1]
        dchi      = (cmax-cmin)/Npnt
        angl      = np.sqrt(2*(1-cost)) if theta>0.005 else theta
        # Do the "outer" integral using the midpoint rule.
        for j1 in range(Npnt):
            rbar = cmin + (j1+0.5)*dchi
            rt   = rbar * theta
            dlnr = np.log(2/angl)/Npnt
            dr   = np.exp( np.log(rbar*angl)+(np.arange(Npnt)+0.5)*dlnr )
            th2  = 2*(1-cost) if theta>0.005 else theta**2
            delt = np.sqrt( (dr**2-rbar**2*th2)/(1+cost)/2 )
            qq   = 2*self.pchi(rbar-delt)*self.pchi(rbar+delt)
            intgd= qq*xir(dr)*dr**2/delt
            dsum+= np.sum(intgd) * dlnr
        dsum *= dchi / (1+cost)
        return(dsum)
        #
    def observe_wp(self,tt):
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
    def observe_wt(self,tt):
        """Converts theory to binned observation."""
        ##ss   = Spline(tt[:,0],tt[:,1])
        xir  = self.extend_xir(tt)
        tfac = np.sqrt(self.xx[1]/self.xx[0])
        thta = np.geomspace(self.xx[0]/tfac,self.xx[-1]*tfac,32)*np.pi/180.
        wth  = Spline(thta,[self.wtheta(t,xir) for t in thta])
        obs  = np.zeros_like(self.xx)
        for i in range(self.xx.size):
            tmin    = self.xx[i] / tfac * np.pi/180.
            tmax    = self.xx[i] * tfac * np.pi/180.
            tval    = np.linspace(tmin,tmax,32)
            obs[i]  = np.trapz(tval*wth(tval),x=tval)
            obs[i] /= np.trapz(tval,x=tval)
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
        thy  = self.observe_wt(tt)
        self.model.d['wth'] = thy.tolist()
        # and compute chi^2.
        chi2 = np.dot(self.dd-thy,np.dot(self.cinv,self.dd-thy))
        chi2+= ( (self.model.nbar-7e-4)/1e-4 )**4
        chi2+= self.prior_chi2(p)
        return(-0.5*chi2)
    #
