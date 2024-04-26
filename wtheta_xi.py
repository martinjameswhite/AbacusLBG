#!/usr/bin/env python3
#
# Computes the angular correlation function, w(theta),
# given a (pre-computed) correlation function and
# redshift distribution.
# This is a simple port of the C code.
#
import numpy       as np
import sys

from scipy.integrate   import simps as simpson
from scipy.special     import legendre
from scipy.special     import hyp2f1    # For D(z).
from scipy.interpolate import InterpolatedUnivariateSpline as Spline



class LCDM:
    # Just some useful distance/redshift functions for LCDM.
    def T_AK(self,x):
        """The "T" function of Adachi & Kasai (2012), used below."""
        b1,b2,b3=2.64086441,0.883044401,0.0531249537
        c1,c2,c3=1.39186078,0.512094674,0.0394382061
        x3   = x**3
        x6   = x3*x3
        x9   = x3*x6
        tmp  = 2+b1*x3+b2*x6+b3*x9
        tmp /= 1+c1*x3+c2*x6+c3*x9
        tmp *= x**0.5
        return(tmp)
        #
    def chi_of_z(self,zz):
        """The comoving distance to redshift zz, in Mpc/h.
           Uses the Pade approximate of Adachi & Kasai (2012) to compute chi
           for a LCDM model, ignoring massive neutrinos."""
        s_ak = (self.OmX/self.OmM)**0.3333333
        tmp  = self.T_AK(s_ak)-self.T_AK(s_ak/(1+zz))
        tmp *= 2997.925/(s_ak*self.OmM)**0.5
        return(tmp)
        #
    def E_of_z(self,zz):
        """The dimensionless Hubble parameter at zz."""
        Ez = (self.OmM*(1+zz)**3 + self.OmX)**0.5
        return(Ez)
        #
    def D_of_z(self,zz):
        """Scale-independent growth factor for flat LCDM."""
        aa = 1./(1.+zz)
        rr = self.OmX/self.OmM
        t1 = hyp2f1(1./3,1,11./6,-aa**3*rr)
        t2 = hyp2f1(1./3,1,11./6,-rr)
        return( aa * t1/t2 )
        #
    def __init__(self,OmM=0.3):
        self.OmM = OmM
        self.OmX = 1-OmM






def read_pchi(fname,cc,Nchi=750):
    """Reads dN/dz and hence computes p(chi).  Returns a spline."""
    dndz = np.loadtxt(fname)
    chi  = cc.chi_of_z(dndz[:,0])
    zchi = Spline(chi,dndz[:,0])
    dnc  = Spline(chi,dndz[:,1])
    cmin = chi[ 0]
    cmax = chi[-1]
    chi  = np.linspace(cmin,cmax,Nchi)
    zchi = zchi(chi)
    dnc  = dnc(chi)
    pchi = dnc*cc.E_of_z(zchi)
    pchi/= simpson(pchi,x=chi)
    pchi = Spline(chi,pchi,ext='zeros')
    return( pchi )
    #




def read_xi(fname):
    """Reads the (real-space) correlation function, returning a spline."""
    xir = np.loadtxt(fname)
    #xir = Spline(xir[:,0],xir[:,1],ext='raise')
    xir = Spline(xir[:,0],xir[:,1],ext='zeros')
    return(xir)
    #



def w_theta(theta,pchi,xir,Npnt=750):
    """Returns w(theta) evaluated at (scalar!) theta, using p(chi) and xi(r).
    Uses Eq.(23) of https://arxiv.org/abs/astro-ph/0609165 ."""
    cost,dsum = np.cos(theta),0.0
    cmin,cmax = pchi.get_knots()[0],pchi.get_knots()[-1]
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
        qq   = 2*pchi(rbar-delt)*pchi(rbar+delt)
        intgd= qq*xir(dr)*dr**2/delt
        dsum+= np.sum(intgd) * dlnr
    dsum *= dchi / (1+cost)
    return(dsum)
    #




if __name__=="__main__":
    if len(sys.argv)!=3:
        raise RuntimeError("Usage: "+sys.argv[0]+" <dndz-fn> <xir-fn>")
    cc   = LCDM()
    pchi = read_pchi(sys.argv[1],cc)
    xir  = read_xi(sys.argv[2])
    # Set the angles in arcminutes.
    thta = [np.exp(np.log(0.5)+(i+0.5)/21*np.log(30./0.5)) for i in range(21)]
    #
    for i in range(len(thta)):
        wth  = w_theta(thta[i]/60*np.pi/180.,pchi,xir)
        print("{:10.4f} {:12.4e}".format(thta[i],wth))
