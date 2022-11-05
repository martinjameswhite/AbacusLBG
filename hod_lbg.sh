#!/bin/bash -l
#SBATCH -J HOD
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o HOD_LBG.out
#SBATCH -e HOD_LBG.err
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A m68
#
module unload craype-hugepages2M
#
source activate abacus
#
date
python lbg_clustering.py
date
#
