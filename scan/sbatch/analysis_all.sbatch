#!/bin/bash
#SBATCH --job-name=analysis_all
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/logs/analysis_all-%j.out

set -uo pipefail
PY=/publicfs10/fs10-m9/home/m9s005258/qy/venv-mace/bin/python
REPO=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/new
SCAN=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "=== analysis_all start $(date) ==="
cd $REPO
$PY -m sampling.scan_analysis --root $SCAN/ana_root --stage all --out $SCAN/traj/analysis_scan
echo "=== analysis_all done $(date) ==="
