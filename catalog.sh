#!/bin/bash
#SBATCH -J Catalog
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -o Catalog.out
#SBATCH -e Catalog.err
#SBATCH -q debug
#SBATCH -C haswell
#SBATCH -A m68
#
source activate abacus
#
echo "Job starting."
#
python3 << EOF
import numpy as np
import time
from abacusnbody.metadata import get_meta
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
#
zz=3.0
dbase=r'/global/cfs/cdirs/desi/cosmosim/Abacus/'
simname=r'AbacusSummit_base_c000_ph000'
meta = get_meta(simname,redshift=zz)
keys = ['n_s', 'omega_b', 'omega_cdm', 'omega_ncdm', 'N_ncdm', 'N_ur',\
        'H0', 'w0', 'wa', 'w', 'Omega_DE', 'Omega_K', 'Omega_M', 'Omega_Smooth', \
        'Redshift', 'ScaleFactor', \
        'OmegaNow_DE', 'OmegaNow_K', 'OmegaNow_m', \
        'f_growth', 'fsmooth', 'Growth', 'Growth_on_a_n', \
        'SimComment', 'SimName', 'SimSet', \
        'BoxSize', 'NP', \
        'BoxSizeHMpc', 'BoxSizeMpc', 'HubbleTimeGyr', 'HubbleTimeHGyr', \
        'ParticleMassHMsun', 'ParticleMassMsun',\
        'InitialRedshift', 'LagrangianPTOrder']
for k in keys:
    print("{:>22s} : ".format(k) + str(meta[k]))
print("\n")
#
flds = ['id','N','x_L2com','v_L2com','sigmav3d_L2com']
cat  = CompaSOHaloCatalog(dbase+simname+'/halos/z{:05.3f}/'.format(zz),fields=flds)
print(cat.halos[:5])
#
npart = cat.halos[:]['N']
mhalo = cat.halos[:]['N'] * meta['ParticleMassHMsun']
print("Lowest mass halo is {:12.4e}Msun/h ({:d} particles).".\
      format(np.min(mhalo[mhalo>0]),np.min(npart[npart>0])))
print(mhalo[:5])
#
print("\n\n\n")
for i in range(16): # 32
    cat  = CompaSOHaloCatalog(dbase+simname+'/halos/z{:05.3f}/'.format(zz),fields=flds)
    print(cat.halos[:25])
    time.sleep(5)
#
#
EOF
#
