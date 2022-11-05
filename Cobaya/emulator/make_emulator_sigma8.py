import numpy as np
import json
import sys
import os
from mpi4py import MPI

from taylor_approximation import compute_derivatives
from compute_class_sigma8 import compute_sigma8


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

basedir = sys.argv[1] + '/'

output_shape = (1,)

# First construct the grid
order = 4
# these are OmegaM, h
x0s = [0.3150, 0.6736]; Nparams = len(x0s) # these are chosen to be roughly at the simulation value
dxs = [0.0100, 0.0100]


template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

X0grid = np.zeros( (Npoints,)*Nparams + output_shape )
X0gridii = np.zeros( (Npoints,)*Nparams + output_shape )

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        X0gridii[iis] = compute_sigma8(coord)

comm.Allreduce(X0gridii, X0grid, op=MPI.SUM)
del(X0gridii)

# Now compute the derivatives
derivs0 = compute_derivatives(X0grid, dxs, center_ii, 5)

# Now save:
if mpi_rank == 0:
    
    # Make the emulator (emu) directory if it
    # doesn't already exist.
    fb = basedir
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    #
    outfile = basedir+'/emu_s8.json'

    list0 = [ dd.tolist() for dd in derivs0 ]

    outdict = {'params': ['omegam', 'h'],\
           'x0': x0s,\
           'lnA0': 3.036394,\
           'derivs0': list0,}

    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()
