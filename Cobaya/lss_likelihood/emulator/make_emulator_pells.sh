#!/bin/bash -l
#SBATCH -J MakePell
#SBATCH -t 0:15:00
#SBATCH -N 1
#SBATCH -o MakePell.out
#SBATCH -e MakePell.err
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -A m68
#
date
#
source activate cmb
#
export PYTHONPATH=${PYTHONPATH}:$SCRATCH/Abacus/Cobaya/BOSSxPlanck/emulator
export OMP_NUM_THREADS=4
#
echo "Setup done.  Starting to run code ..."
#
# Now run the code.
#
Omfid=0.315
#
for zeff in 3.00 ; do
  srun -n 16 -c 4 python make_emulator_pells.py $PWD $zeff $Omfid
done
#
date
#
