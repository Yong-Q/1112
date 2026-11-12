#!/bin/bash
#SBATCH --job-name=bench_timing
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/logs/bench_timing-%j.out

source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
LMP=/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp
cd /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/bench/timing
echo "=== TIMING BENCH start $(date) node=$(hostname) ==="

echo "--- A_rho002_timing ---"
rm -f log.A_rho002_timing
mpirun --allow-run-as-root --oversubscribe -np 32 $LMP -k on t 1 -in in.A_rho002_timing -log log.A_rho002_timing -screen none
grep -A6 "^Loop time" log.A_rho002_timing | grep -E "^(Loop time|Pair|Neigh|Comm)"
echo "  [timing] $(grep -E "^ *300 +[0-9]" log.A_rho002_timing | tail -1)"

echo "--- B_rho0237_timing ---"
rm -f log.B_rho0237_timing
mpirun --allow-run-as-root --oversubscribe -np 32 $LMP -k on t 1 -in in.B_rho0237_timing -log log.B_rho0237_timing -screen none
grep -A6 "^Loop time" log.B_rho0237_timing | grep -E "^(Loop time|Pair|Neigh|Comm)"
echo "  [timing] $(grep -E "^ *300 +[0-9]" log.B_rho0237_timing | tail -1)"

echo "--- C_rho002_skin4 ---"
rm -f log.C_rho002_skin4
mpirun --allow-run-as-root --oversubscribe -np 32 $LMP -k on t 1 -in in.C_rho002_skin4 -log log.C_rho002_skin4 -screen none
grep -A6 "^Loop time" log.C_rho002_skin4 | grep -E "^(Loop time|Pair|Neigh|Comm)"
echo "  [timing] $(grep -E "^ *300 +[0-9]" log.C_rho002_skin4 | tail -1)"

echo "--- D_rho002_materialized ---"
rm -f log.D_rho002_materialized
mpirun --allow-run-as-root --oversubscribe -np 32 $LMP -k on t 1 -in in.D_rho002_materialized -log log.D_rho002_materialized -screen none
grep -A6 "^Loop time" log.D_rho002_materialized | grep -E "^(Loop time|Pair|Neigh|Comm)"
echo "  [timing] $(grep -E "^ *300 +[0-9]" log.D_rho002_materialized | tail -1)"

echo "--- E_rho002_speed ---"
rm -f log.E_rho002_speed
mpirun --allow-run-as-root --oversubscribe -np 32 $LMP -k on t 1 -in in.E_rho002_speed -log log.E_rho002_speed -screen none
grep -A6 "^Loop time" log.E_rho002_speed | grep -E "^(Loop time|Pair|Neigh|Comm)"
echo "  [timing] $(grep -E "^ *300 +[0-9]" log.E_rho002_speed | tail -1)"

echo "=== TIMING BENCH done $(date) ==="
