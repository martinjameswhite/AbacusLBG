import numpy as np
import json
import sys
import os
from mpi4py import MPI

from compute_fid_dists import compute_fid_dists
from taylor_approximation import compute_derivatives


comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
if mpi_rank==0:
    print(sys.argv[0]+" running on {:d} processes.".format(mpi_size))
#print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

basedir = sys.argv[1] +'/'
z = float(sys.argv[2])
Omfid = float(sys.argv[3])

# Compute fiducial distances
fid_dists = compute_fid_dists(z,Omfid)

# Set up the output k vector:
from compute_pell_tables import compute_pell_tables, kvec

output_shape = (len(kvec),19) # two multipoles and 19 types of terms


# First construct the grid

order = 4
# these are OmegaM, h, sigma8
x0s = [0.3150, 0.6736, 0.81]; Nparams = len(x0s) # these are chosen to be roughly at the simulation value
dxs = [0.0100, 0.0100, 0.05]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')

P0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P2grid = np.zeros( (Npoints,)*Nparams+ output_shape)
P4grid = np.zeros( (Npoints,)*Nparams+ output_shape)

P0gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P2gridii = np.zeros( (Npoints,)*Nparams+ output_shape)
P4gridii = np.zeros( (Npoints,)*Nparams+ output_shape)

for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        p0, p2, p4 = compute_pell_tables(coord,z=z,fid_dists=fid_dists)
        
        P0gridii[iis] = p0
        P2gridii[iis] = p2
        P4gridii[iis] = p4
        
comm.Allreduce(P0gridii, P0grid, op=MPI.SUM)
comm.Allreduce(P2gridii, P2grid, op=MPI.SUM)
comm.Allreduce(P4gridii, P4grid, op=MPI.SUM)

del(P0gridii, P2gridii, P4gridii)

# Now compute the derivatives
derivs0 = compute_derivatives(P0grid, dxs, center_ii, 5)
derivs2 = compute_derivatives(P2grid, dxs, center_ii, 5)
derivs4 = compute_derivatives(P4grid, dxs, center_ii, 5)

if mpi_rank == 0:
    # Make the emulator directory if it
    # doesn't already exist.
    fb = basedir
    if not os.path.isdir(fb):
        print("Making directory ",fb)
        os.mkdir(fb)
    else:
        print("Found directory ",fb)
    #
comm.Barrier()

# Now save:
outfile = basedir + '/emu_z%.2f_pkells.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]
list4 = [ dd.tolist() for dd in derivs4 ]

outdict = {'params': ['omegam', 'h', 'sigma8'],\
           'x0': x0s,\
           'kvec': kvec.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,\
           'derivs4': list4}

if mpi_rank == 0:
    json_file = open(outfile, 'w')
    json.dump(outdict, json_file)
    json_file.close()
