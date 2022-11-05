#!/bin/bash
#SBATCH -J LCDM
#SBATCH -N 1
#SBATCH -t 0:30:00
#SBATCH -o Cobaya.out
#SBATCH -e Cobaya.err
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A m68
#
source activate cobaya
#
export PYTHONPATH=${PYTHONPATH}:$PWD/lss_likelihood
export PYTHONPATH=${PYTHONPATH}:$PWD/emulator
export OMP_NUM_THREADS=2
export NUMEXPR_MAX_THREADS=2
#
fb=lbg_z300_r245_rsd_am
#
echo "Job starting."
#
# To start a run:
rm -rf chains/${fb}.*
srun --cpu-bind=cores -n 16 -c 2 cobaya-run ${fb}.yaml
# To restart:
#srun --cpu-bind=cores -n 16 -c 2 cobaya-run chains/${fb}
#
