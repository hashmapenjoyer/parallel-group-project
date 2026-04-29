#!/bin/bash
# Smoke test: 1 node, 1 rank, small grid. Use to confirm the binary
# launches before queueing big jobs.
#
#   sbatch scripts/run_debug.sh

#SBATCH --job-name=gol_debug
#SBATCH --partition=dcs-2024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=bench/results/slurm-%j.out

RCS_ID=PCPGbrnr
PROJ=/gpfs/u/home/PCPG/${RCS_ID}/scratch/parallel-group-project

module load xl_r spectrum-mpi cuda/11.2
cd $PROJ
mkdir -p bench/results

mpirun -gpu -n 1 \
    ./bin/gol_mpi_cuda --w 8192 --h 8192 --steps 1000 \
        --mode gpu --kernel naive --label debug
