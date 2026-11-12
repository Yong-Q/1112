#!/bin/bash
#SBATCH --job-name=bench_co2
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/logs/bench_co2-%j.out

source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
LMP=/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp
cd /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/bench/co2
echo "=== BENCH start $(date) node=$(hostname) ==="

echo "--- np4 np=4 ---"
rm -f log.np4
mpirun -np 4 $LMP -k on -in in.np4 -log log.np4 -screen none && \
  grep -A6 "^Loop time" log.np4 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np4"

echo "--- np8 np=8 ---"
rm -f log.np8
mpirun -np 8 $LMP -k on -in in.np8 -log log.np8 -screen none && \
  grep -A6 "^Loop time" log.np8 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np8"

echo "--- np16 np=16 ---"
rm -f log.np16
mpirun -np 16 $LMP -k on -in in.np16 -log log.np16 -screen none && \
  grep -A6 "^Loop time" log.np16 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np16"

echo "--- np32 np=32 ---"
rm -f log.np32
mpirun -np 32 $LMP -k on -in in.np32 -log log.np32 -screen none && \
  grep -A6 "^Loop time" log.np32 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np32"

echo "--- np32_skin4 np=32 ---"
rm -f log.np32_skin4
mpirun -np 32 $LMP -k on -in in.np32_skin4 -log log.np32_skin4 -screen none && \
  grep -A6 "^Loop time" log.np32_skin4 | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np32_skin4"

echo "--- np32_mat np=32 ---"
rm -f log.np32_mat
mpirun -np 32 $LMP -k on -in in.np32_mat -log log.np32_mat -screen none && \
  grep -A6 "^Loop time" log.np32_mat | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np32_mat"

echo "--- np32_speed np=32 ---"
rm -f log.np32_speed
mpirun -np 32 $LMP -k on -in in.np32_speed -log log.np32_speed -screen none && \
  grep -A6 "^Loop time" log.np32_speed | grep -E "^(Loop time|Pair|Neigh|Comm)" || echo "FAILED: np32_speed"

echo "=== BENCH done $(date) ==="
