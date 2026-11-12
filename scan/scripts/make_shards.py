#!/usr/bin/env python3
"""Split the 5-molecule density manifest into 4 shards per molecule and write
sbatch scripts (32 cores for H2O/NH3/CH4, 48 cores for C2H4/C2H6)."""
from __future__ import annotations
import csv
from pathlib import Path

SCAN = Path("/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan")
REPO = "/publicfs10/fs10-m9/home/m9s005258/qy/cdft/new"
PY = "/publicfs10/fs10-m9/home/m9s005258/qy/venv-mace/bin/python"
ART = ("/publicfs10/fs10-m9/home/m9s005258/.cache/symmetrix/execution-jit/"
       "artifacts/8d00f0566cc8a5f6a603befb0eab51f47d3a025c19c9b9d92e0731eb351e84f7/"
       "factorized_host_plugin_gen11_src0c65a7c611c15feff220748dc034a7a963176ec834be421668b0f9c04a5ca982.so")
MODEL = "/publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/maceles_off_chno.json"
CUBIN = "/publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/maceles_off.cubin"
LMP = "/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp"
LDLP = ("/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:"
        "/publicfs10/fs10-share/soft/share-soft/openmpi/openmpi-4.1.8/install-new/lib:"
        "/publicfs10/fs10-share/soft/share-soft/gcc/gcc-14.2.0/install/lib64")

# molecule -> (cores, mem, time)
RES = {
    "H2O":  (32, "32G",  "60-00:00:00"),
    "NH3":  (32, "32G",  "60-00:00:00"),
    "CH4":  (32, "48G",  "60-00:00:00"),
    "C2H4": (48, "64G",  "60-00:00:00"),
    "C2H6": (48, "64G",  "60-00:00:00"),
}
SPLIT = [3, 3, 2, 2]

COLUMNS = ["sim_id", "data", "traj", "species", "temperature", "timestep",
           "steps", "dump_every", "density", "c2_pair", "seed", "elements"]


def sbatch_text(mol: str, k: int, cores: int, mem: str, time: str) -> str:
    shard = f"shard_{mol}_{k}"
    job = f"cdft_{mol}_{k}"
    out = f"{SCAN}/logs/{shard}-%j.out"
    return f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=amd_m9_768
#SBATCH --account=bscc-m9
#SBATCH --nodes=1
#SBATCH --ntasks={cores}
#SBATCH --cpus-per-task=1
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={out}

set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH

echo "=== {shard} === node=$(hostname) ntasks=$SLURM_NTASKS ==="
cd {REPO}

echo "--- MD (no-analyze, resumable) ---"
{PY} sampling/les_batch.py \\
  --manifest {SCAN}/{shard}/attempt-01/manifest.csv \\
  --outroot {SCAN}/traj \\
  --run --resume --no-analyze --no-compress-traj \\
  --gpus 0 --np-ranks {cores} --omp-threads 1 \\
  --model {MODEL} \\
  --cubin {CUBIN} \\
  --host-artifact {ART} \\
  --lmp {LMP} \\
  --ld-library-path "{LDLP}"

echo "--- post-process ---"
{PY} {SCAN}/../scripts/postprocess_shard.py \\
  --manifest {SCAN}/{shard}/attempt-01/manifest.csv \\
  --traj-root {SCAN}/traj --post-root {SCAN}/post

echo "=== {shard} done ==="
"""


def main() -> int:
    manifest = SCAN / "density_scan_5mol.csv"
    rows = list(csv.DictReader(open(manifest)))
    by_species: dict[str, list[dict]] = {}
    for r in rows:
        by_species.setdefault(r["species"], []).append(r)
    total_shards = 0
    for mol, mol_rows in by_species.items():
        assert len(mol_rows) == 10, (mol, len(mol_rows))
        cores, mem, time = RES[mol]
        idx = 0
        for k, size in enumerate(SPLIT, 1):
            chunk = mol_rows[idx:idx + size]
            idx += size
            shard_dir = SCAN / f"shard_{mol}_{k}" / "attempt-01"
            shard_dir.mkdir(parents=True, exist_ok=True)
            with open(shard_dir / "manifest.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(chunk)
            (shard_dir / "run.sbatch").write_text(
                sbatch_text(mol, k, cores, mem, time))
            rhos = " ".join(r["sim_id"].rsplit("rho", 1)[-1] for r in chunk)
            print(f"shard_{mol}_{k}: {cores}c {mem} {time}  [{len(chunk)}] {rhos}")
            total_shards += 1
    print(f"wrote {total_shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
