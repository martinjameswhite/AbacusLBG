#!/usr/bin/env python
#
# Python implementation of an affine, ensemble sampler for running
# Markov chains.
# This code uses MPI, with one likelihood evaluation (from an external
# package) per MPI process.  Each process is assumed to have Nthread
# OpenMP threads available for running codes which can take advantage
# of shared-memory parallelism in a hybrid computing environment.
#

__author__ = "Martin White"
__version__ = "1.0"
__email__  = "mwhite@berkeley.edu"


# Need to work around some Python path issues on Cori.
import sys
import os
#
import numpy as np
from hod_likelihood import Likelihood
from mpi4py         import MPI


comm = MPI.COMM_WORLD
me   = comm.Get_rank()
nproc= comm.Get_size()


# Probably want to move a bunch of this into a class.

def draw_gz():
    """Returns a random z from g(z)~1/sqrt(z) in [1/a,a].
    This can be replaced by something more clever later."""
    sqrta = 2.0**0.5
    z     = ((sqrta-1./sqrta)*np.random.uniform()+1/sqrta)**2
    return(z)
    #




def stretch_move(comm,par,lnLik):
    """Does a single "stretch move" step for the affine sampler.
    The communicator class is passed in, and the parameters and
    previous lnLikelihood are stored in (nwalker,npar) and (nwalker)
    NumPy arrays.  Updates are returned as a tuple of NumPy arrays."""
    # This is done in two steps.
    for i in range(2):
        k    = me + i*nproc				# The walker to update
        j    = np.random.randint(0,nproc)+(1-i)*nproc	# The partner to k.
        z    = draw_gz()
        y    = par[j,:]+z*(par[k,:]-par[j,:])
        lnLy = lik(y)					# Time intensive step.
        if lnLy-lnLik[k]<150:
            q= np.exp( (par.shape[1]-1)*np.log(z) + lnLy-lnLik[k])
            if (lnLy-lnLik[k]<-150)|(np.random.uniform()>=q):
                # Reject this step by overwriting trial values.
                y    = par[k,:]
                lnLy = lnLik[k]
            else:
                # Keeping the values.
                if lnLy<-5000:
                    ostr = "******************\n" +\
                           "proc={:4d}, k={:4d}, partner {:4d}, z={:e}\n".format(me,k,j,z) +\
                           "y="+str(y) + "\n" +\
                           "Keeping lnL="+str(lnLy)+" over lnL[k]={:e}\n\n".format(lnLik[k])
                    print(ostr,flush=True)
        # Gather the results from the nproc checks we have just done.
        rpar = np.zeros(nproc*y.size,dtype='float')
        rLik = np.zeros(nproc,dtype='float')
        comm.Allgather(   y,rpar)
        comm.Allgather(lnLy,rLik)
        rpar.shape=(nproc,y.size)
        # and use them to pack the full nwalker-sized arrays.
        if i==0:	# Simply make a copy.
            par[:nproc,:]= rpar.copy()
            lnLik[:nproc]= rLik.copy()
            gpar = rpar.copy()
            gLik = rLik.copy()
        else:		# append, so walkers nproc..nwalker are filled in.
            gpar = np.concatenate( (gpar,rpar) )
            gLik = np.concatenate( (gLik,rLik) )
    return( (gpar,gLik) )
    #





def sample(fout,lik,Nstep=500):
    """This runs the code.
    lik is a likelihood instance.
    To avoid buffering/flushing issues the output is written (by task 0)
    to the file "fout"."""
    #
    outstr = "Process {:4d} of {:4d} starting.".format(me,nproc)
    print(outstr,flush=True)
    comm.Barrier()
    # Get the starting position and scatter.
    initpar,sigma = lik.startpar()
    #
    npar    = len(initpar)
    nwalker = 2 * nproc
    # Initialize the parameters for each process.
    # If an old chain exists, use the last entries of that, otherwise
    # use initpar and sigma.
    if os.path.isfile(fout):
        if me==0:
            print("Starting from old chain "+fout,flush=True)
            chain = np.loadtxt(fout)
            nend  = chain.shape[0] - nwalker
            if nend<0:
                 print("Old chain too short!",flush=True)
                 comm.Abort(1)
            par   = np.ascontiguousarray(chain[nend:,:-1])
            chain = None
        else:
            par   = np.zeros( (nwalker,npar) )
    else:
        par = np.zeros( (nwalker,npar),dtype='float' )
        for j in range(nwalker):
            mfac     = (1 + sigma*np.random.normal(size=npar)).clip(0.15,1.85)
            par[j,:] = initpar*mfac
    comm.Bcast(par,root=0)
    # Now compute initial likelihoods -- we redo this even if starting
    # from a previous chain in case priors or settings have changed.
    for i in range(2):
        lnL0 = lik(par[me+i*nproc,:])	# Time intensive step.
        rLik = np.zeros(nproc,dtype='float')
        comm.Allgather(lnL0,rLik)
        if i==0:	# Simply make a copy.
            lnLik = rLik.copy()
        else:	# append, so walkers nproc..nwalker are filled in.
            lnLik = np.concatenate( (lnLik,rLik) )
    # Finally generate the steps in the chain.
    np.random.seed(1234+me)
    if me==0:
        fp = open(fout[:-4]+"head","w")
        fp.write(lik.header)
        fp.close()
        fp = open(fout,"w")
    for i in range(Nstep):	# Or some stopping condition.
        par,lnLik = stretch_move(comm,par,lnLik)
        if me==0:
            for j in range(nwalker):
                ss = ""
                for k in range(npar):
                    ss += " {:12.6f}".format(par[j,k])
                fp.write(ss+" {:12.4e}\n".format(lnLik[j]))
            fp.flush()
    if me==0:
        fp.close()





if __name__=="__main__":
    if len(sys.argv)!=1:
        outstr = "Usage: "+sys.argv[0]
        raise RuntimeError(outstr)
    # Set up a likelihood instance and output file name.
    lik  = Likelihood(MPI.COMM_WORLD.Get_rank())
    fout = "hod_fit.mcmc"
    # and call the sampler:
    sample(fout,lik,10000)
    #
