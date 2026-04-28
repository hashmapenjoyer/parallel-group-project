#!/bin/bash
# Interactive-debug-queue smoke test: 1 node, 1 rank, small grid.
# Use this to confirm the binary launches before submitting big jobs.
#
#   sbatch scripts/run_debug.sh

#SBATCH --partition=dcs
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=gol_debug

RCS_ID=PCPGbrnr
PROJ=/gpfs/u/scratch/${RCS_ID}/parallel-group-project

module load xl_r spectrum-mpi cuda/11.2
cd $PROJ

jsrun -n 1 -a 1 -c 4 -g 1 \
    ./bin/gol_mpi_cuda --w 1024 --h 1024 --steps 100 \
        --mode gpu --kernel naive --label debug
