#!/bin/bash -l
#SBATCH -J MakeSig8
#SBATCH -t 0:05:00
#SBATCH -N 1
#SBATCH -o MakeSig8.out
#SBATCH -e MakeSig8.err
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -A m68
#
date
#
source activate cmb
#
export PYTHONPATH=${PYTHONPATH}:$PWD
export OMP_NUM_THREADS=4
#
echo "Setup done.  Starting to run code ..."
#
# Now run the code.
#
srun -n 16 -c 4 python make_emulator_sigma8.py $PWD
#
date
#
