#!/bin/bash
#SBATCH --job-name=cdft_smoke_mpi
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/_smoke/sbatch_mpi-%j.out
set -e
source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH
echo "NODE=$(hostname) SLURM_NTASKS=$SLURM_NTASKS SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"
which mpirun
mpirun --version | head -2
mpirun --allow-run-as-root -np 8 hostname | sort | uniq -c
