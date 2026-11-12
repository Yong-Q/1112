#!/bin/bash
#SBATCH --job-name=cdft_H2O_1
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=60-00:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/logs/shard_H2O_1-%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH

echo "=== shard_H2O_1 === node=$(hostname) ntasks=$SLURM_NTASKS ==="
cd /publicfs10/fs10-m9/home/m9s005258/qy/cdft/new

echo "--- MD (no-analyze, resumable) ---"
/publicfs10/fs10-m9/home/m9s005258/qy/venv-mace/bin/python sampling/les_batch.py \
  --manifest /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/shard_H2O_1/attempt-01/manifest.csv \
  --outroot /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj \
  --run --resume --no-analyze --no-compress-traj \
  --gpus 0 --np-ranks 32 --omp-threads 1 \
  --model /publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/maceles_off_chno.json \
  --cubin /publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/maceles_off.cubin \
  --host-artifact /publicfs10/fs10-m9/home/m9s005258/.cache/symmetrix/execution-jit/artifacts/8d00f0566cc8a5f6a603befb0eab51f47d3a025c19c9b9d92e0731eb351e84f7/factorized_host_plugin_gen11_src0c65a7c611c15feff220748dc034a7a963176ec834be421668b0f9c04a5ca982.so \
  --lmp /publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp \
  --ld-library-path "/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:/publicfs10/fs10-share/soft/share-soft/openmpi/openmpi-4.1.8/install-new/lib:/publicfs10/fs10-share/soft/share-soft/gcc/gcc-14.2.0/install/lib64"

echo "--- post-process ---"
/publicfs10/fs10-m9/home/m9s005258/qy/venv-mace/bin/python /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/../scripts/postprocess_shard.py \
  --manifest /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/shard_H2O_1/attempt-01/manifest.csv \
  --traj-root /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj --post-root /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/post

echo "=== shard_H2O_1 done ==="
