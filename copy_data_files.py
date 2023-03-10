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
# Copy the ancillary targeting file.
db1=/global/cfs/cdirs/desi/survey/fiberassign/special/tertiary
db2=/0015/inputcats/
fn=lbg-xmmlss-fall2022-hipr24.2-extsub1.0.fits
cp ${db1}${db2}${fn} clauds-xmmlss-cat.fits
