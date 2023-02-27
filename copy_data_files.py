#!/bin/bash
# Set the path.
db=/global/cfs/cdirs/desi/users/raichoor/laelbg/clauds
# Copy the mask files.
cp ${db}/clauds-cosmos-hpmask.fits .
cp ${db}/clauds-xmmlss-hpmask.fits .
# and the random catalog.
cp ${db}/clauds-cosmos-rands-dens10000.fits .
cp ${db}/clauds-xmmlss-rands-dens10000.fits .
#
