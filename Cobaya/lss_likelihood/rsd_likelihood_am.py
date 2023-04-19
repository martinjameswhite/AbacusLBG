import numpy as np
import json
import yaml
#
from cobaya.theory        import Theory
from cobaya.likelihood    import Likelihood
from scipy.interpolate    import InterpolatedUnivariateSpline as Spline
#
from taylor_approximation import taylor_approximate
from compute_sigma8_class import Compute_Sigma8



# Class for a full-shape likelihood.
class FullShapeLikelihood(Likelihood):
    zfid:      float
    Hz_fid:    float
    chiz_fid:  float
    #
    basedir:   str
    fs_datfn:  str
    fs_covfn:  str
    fs_winfn:  str
    lp_fname:  str
    #
    fs_kmin:   str
    fs_mmax:   str
    fs_qmax:   str
    #
    def initialize(self):
        """Sets up the class."""
        self.niter = 0
        # Load the data, window function matrix and covariance matrix.
        self.loadData()
        # Handle the linear parameters.
        self.linpar = yaml.load(open(self.basedir+self.lp_fname),Loader=yaml.SafeLoader)
        self.lp_avg = {k: float(self.linpar[k]['avg']) for k in self.linpar.keys()}
        self.lp_std = {k: float(self.linpar[k]['std']) for k in self.linpar.keys()}
        self.Nlin   = len(self.linpar)
        #
    def get_requirements(self):
        req = {'taylor_pk_ell_mod': None,\
               'H0': None,\
               'sigma8': None,\
               'omegam': None,\
               'logA': None,\
               'bsig8': None,\
               'b2': None,\
               'bs': None,\
               'b3': None}
        return(req)
        #
    def logp(self,**params_values):
        """Return a log-likelihood."""
        # First the prediction with linear parameters at their means.
        thetas  = self.lp_avg.copy()
        fs_thy  = self.fs_predict(thetas)
        fs_obs  = self.fs_observe(fs_thy)
        self.obs= fs_obs
        diff    = self.dd - fs_obs
        # Compute the templates just by finite difference, knowing the
        # parameter dependence is linear.
        self.templates,ivar = [],[]
        for par in self.linpar.keys():
            thetas          = self.lp_avg.copy()
            thetas[par]    += 1.0
            self.templates += [ self.fs_observe(self.fs_predict(thetas))-fs_obs ]
            ivar           += [ 1.0/self.lp_std[par]**2 ]
        self.templates = np.array(self.templates)
        # Now the covariance and offsets from the analytic marginalization.
        TCinv= np.dot(self.templates,self.cinv)
        V    = np.dot(TCinv,diff)
        L    = np.dot(TCinv,self.templates.T)+np.diag(np.array(ivar))
        Linv = np.linalg.inv(L)
        #
        chi2 = np.dot(diff,np.dot(self.cinv,diff))
        chi2-= np.dot(V,np.dot(Linv,V))
        chi2+= np.log(np.linalg.det(L)) - self.Nlin*np.log(2*np.pi)
        # Put in an ad-hoc annealing schedule.
        if self.niter<0:
            self.niter += 1
            coolby      = np.min([30.,1+1000./self.niter])
            chi2       /= coolby
        return(-0.5*chi2)
        #
    def loadData(self):
        """
        Loads the required data.
        The covariance is assumed to already be joint in the concatenated format.
        """
        # First load the data
        fs_dat     = np.loadtxt(self.basedir+self.fs_datfn)
        self.kdat  = fs_dat[:,0]
        self.p0dat = fs_dat[:,1]
        self.p2dat = fs_dat[:,2]
        # Join the data vectors together
        self.dd = np.concatenate( (self.p0dat,self.p2dat) )
        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.fs_covfn)
        # Finally load the window function matrix.
        self.matW = np.loadtxt(self.basedir+self.fs_winfn)
        #
        # We're only going to want some of the entries in computing chi^2, handle
        # this by adjusting Cov.
        startii = 0
        kcut = (self.kdat > self.fs_mmax)\
             | (self.kdat < self.fs_kmin)
        for i in np.nonzero(kcut)[0]:     # FS Monopole.
            ii = i + startii
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e25
        #
        startii += self.kdat.size
        kcut = (self.kdat > self.fs_qmax)\
             | (self.kdat < self.fs_kmin)
        for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
            ii = i + startii
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e25
        # Copy cov and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec
        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])
        #
        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        return p0, p2, p4
        #
    def fs_predict(self,thetas):
        """Use the PT model to compute P_ell, given biases etc."""
        pp   = self.provider
        #
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs
        # The cosmological parameters.
        sig8 = pp.get_param('sigma8')
        #sig8 = pp.get_result('sigma8')
        b1   = pp.get_param('bsig8')/sig8 - 1.
        b2   = pp.get_param('b2')
        bs   = pp.get_param('bs')
        b3   = pp.get_param('b3')
        # the "linear" parameters.
        alp0 = thetas['alpha0']
        alp2 = thetas['alpha2']
        sn0  = thetas['SN0']
        sn2  = thetas['SN2']
        #
        bias  = [b1,b2,bs,b3]
        cterm = [alp0,alp2,0,0]
        stoch = [sn0,sn2,0]
        bvec  = bias + cterm + stoch
        #
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec,p0ktable,p2ktable,p4ktable)
        #
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.0,],kv),np.append([0.0,],p0)
        p2 = np.append([0.0,],p2)
        p4 = np.append([0.0,],p4)
        tt = np.array([kv,p0,p2,p4]).T
        #
        if np.any(np.isnan(tt)):
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        return(tt)
        #
    def fs_observe(self,tt):
        """Apply the window function matrix to get the binned prediction."""
        if True:
            # Skip the window function convolution for now... and just
            # evaluate the theory at the data point k-values.
            kv = self.kdat
            thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
            thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
            self.pconv = thy
            return(thy)
        # Have to stack ell=0 & 2 in bins of 0.001h/Mpc from 0-1.0h/Mpc.
        kv  = np.linspace(0.0,1.0,1000,endpoint=False) + 0.0005
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        #
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            hub = self.provider.get_param('H0') / 100.
            sig8 = self.provider.get_param('sigma8')
            OmM = self.provider.get_param('omegam')
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        #
        # Convolve with window.
        convolved_model = np.matmul(self.matW,thy)
        self.pconv = convolved_model
        return convolved_model

    
    

class Taylor_pk_theory(Theory):
    """
    A class to return a set of tables from the Taylor series of Pkell.
    """
    zfid:        float
    pk_filename: str
    s8_filename: str
    basedir:     str
    #
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        # First Load Sigma8 class:
        self.compute_sigma8 = Compute_Sigma8(self.basedir + self.s8_filename)
        # 
        # Load the power spectrum derivatives
        json_file = open(self.basedir+self.pk_filename, 'r')
        emu = json.load( json_file )
        json_file.close()
        #    
        x0s  = np.array(emu['x0'])
        kvec = np.array(emu['kvec'])
        derivs_p0 = [np.array(ll) for ll in emu['derivs0']]
        derivs_p2 = [np.array(ll) for ll in emu['derivs2']]
        derivs_p4 = [np.array(ll) for ll in emu['derivs4']]
        #
        # and save them to a dictionary.
        taylor_pk = {}
        taylor_pk['x0'       ] = x0s
        taylor_pk['kvec'     ] = kvec
        taylor_pk['derivs_p0'] = derivs_p0
        taylor_pk['derivs_p2'] = derivs_p2
        taylor_pk['derivs_p4'] = derivs_p4
        self.taylor_pk         = taylor_pk
        del emu
        #
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        #zg  = np.linspace(0,self.zdif,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'omegam': None,\
               'H0': None,\
               'logA': None,\
              }
        return(req)
        #
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod']
        #
    def get_can_provide_params(self):
        return ['sigma8']
        #
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives.
        """
        pp   = self.provider
        hub  = pp.get_param('H0') / 100.
        logA = pp.get_param('logA')
        OmM  = pp.get_param('omegam')
        sig8 = self.compute_sigma8.compute_sigma8(OmM,hub,logA)
        cosmopars = [OmM, hub, sig8]
        #
        # Generate pktables
        x0s     = self.taylor_pk['x0']
        derivs0 = self.taylor_pk['derivs_p0']
        derivs2 = self.taylor_pk['derivs_p2']
        derivs4 = self.taylor_pk['derivs_p4']
        #
        kv = self.taylor_pk['kvec']
        # The order should be <= the order computed by the emulator generator.
        # Higher order is more accurate, but slower.
        p0ktable = taylor_approximate(cosmopars,x0s,derivs0,order=3)
        p2ktable = taylor_approximate(cosmopars,x0s,derivs2,order=3)
        p4ktable = taylor_approximate(cosmopars,x0s,derivs4,order=3)
        ptables  = (kv, p0ktable, p2ktable, p4ktable)
        #
        state['derived'] = {'sigma8': sig8}
        state['taylor_pk_ell_mod'] = ptables
        #
