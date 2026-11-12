#!/bin/bash
#SBATCH --job-name=post_all
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/logs/post_all-%j.out

set -uo pipefail
PY=/publicfs10/fs10-m9/home/m9s005258/qy/venv-mace/bin/python
SCAN=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "=== post_all start $(date) node=$(hostname) ==="
for m in shard_5 shard_6 shard_7 shard_8 \
         shard_H2O_1 shard_H2O_2 shard_H2O_3 shard_H2O_4 \
         shard_NH3_1 shard_NH3_2 shard_NH3_3 shard_NH3_4 \
         shard_CH4_1 shard_CH4_2 shard_CH4_3 \
         shard_C2H4_1 shard_C2H4_2 shard_C2H4_3 \
         shard_C2H6_1 shard_C2H6_2 shard_C2H6_3 shard_C2H6_4 ; do
  echo "--- $m $(date +%H:%M:%S) ---"
  $PY /publicfs10/fs10-m9/home/m9s005258/qy/cdft/scripts/postprocess_shard.py \
      --manifest $SCAN/$m/attempt-01/manifest.csv \
      --traj-root $SCAN/traj --post-root $SCAN/post || echo "  [WARN] $m 失败"
done
echo "=== post_all done $(date) ==="
