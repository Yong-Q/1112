#!/bin/bash
#SBATCH --job-name=bench_c2h6
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/logs/bench_c2h6-%j.out

source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
LMP=/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp
cd /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/bench/c2h6
echo "=== BENCH start $(date) node=$(hostname) ==="

echo "--- np8 np=8 ---"
rm -f log.np8
mpirun -np 8 $LMP -k on -in in.np8 -log log.np8 -screen none && \
  grep -A6 "^Loop time" log.np8 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np8"

echo "--- np16 np=16 ---"
rm -f log.np16
mpirun -np 16 $LMP -k on -in in.np16 -log log.np16 -screen none && \
  grep -A6 "^Loop time" log.np16 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np16"

echo "--- np24 np=24 ---"
rm -f log.np24
mpirun -np 24 $LMP -k on -in in.np24 -log log.np24 -screen none && \
  grep -A6 "^Loop time" log.np24 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np24"

echo "--- np48 np=48 ---"
rm -f log.np48
mpirun -np 48 $LMP -k on -in in.np48 -log log.np48 -screen none && \
  grep -A6 "^Loop time" log.np48 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np48"

echo "=== BENCH done $(date) ==="
