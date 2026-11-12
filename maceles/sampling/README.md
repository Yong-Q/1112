# LES (MACELES) -> c1/c2 driver

`les_c1c2.py` connects the fixed Symmetrix-MACELES LAMMPS engine
(`pair_style symmetrix/mace/float32/kk ... jit_device_artifact`) to the
existing cDFT extraction pipeline in `cdft_dataset.py`. All correlation
functions are computed by `cdft_dataset.process_bulk`; this driver only
handles MD setup, timing, and output layout.

## Modes

* `--run`: write a LAMMPS input, run NVT, time the run, then extract c1/c2.
* `--analyze-only`: skip MD; extract c1/c2 from an existing dump.

Both write `<outdir>/f1/`, `<outdir>/f2/` and a `<outdir>/les_c1c2.json`
manifest (engine hashes, timing, extraction record).

## Environment

Use the Python that has ASE (e.g. `matos-compute`). The driver sets
`OMP_NUM_THREADS=1` and the engine `LD_LIBRARY_PATH` by default; override
with `--omp-threads` / `--ld-library-path`.

```bash
PY=/opt/matos-tools/conda/envs/matos-compute/bin/python
```

## Run the six 3000-atom species

All six boxes under
`/workspace/group/shared/bza/symmetrix-polar/bench/q4_maceles_off/six_3000/`
use LAMMPS types C=1 H=2 N=3 O=4, so `--elements "C H N O"` is always right.

```bash
$PY sampling/les_c1c2.py --run \
  --data  .../six_3000/H2O/data.lammps \
  --species H2O --elements "C H N O" \
  --temperature 300 --timestep 0.0005 \
  --steps 20000 --dump-every 1000 \
  --c2-pair "O O" \
  --outdir  .../c1c2/H2O \
  --workdir .../runs/H2O
```

| species | molecules | rho (molecules/A^3) | suggested `--c2-pair` |
|---------|-----------|---------------------|------------------------|
| H2O     | 1000      | 0.0334              | `O O`  |
| NH3     | 750       | 0.0241              | `N N`  |
| CH4     | 600       | 0.0158              | `C C`  |
| C2H4    | 500       | 0.0123              | `C C`  |
| C2H6    | 375       | 0.0105              | `C C`  |
| CO2     | 1000      | 0.0137              | `C O`  |

`--density` is in molecules/A^3 and is derived from the data file when
omitted. `--c2-pair` accepts either `"O O"` or `O O`. For long trajectories
use `--stride N` to subsample frames when computing the RDF (default 1).

## Reuse an existing trajectory (e.g. the 100 ps run)

```bash
$PY sampling/les_c1c2.py --analyze-only \
  --traj /workspace/group/shared/bza/symmetrix-polar/bench/q4_maceles_off/h2o_100ps/attempt-01/traj.dump \
  --species H2O --elements "C H N O" \
  --T 300 --density 0.0334 --c2-pair "O O" \
  --outdir .../c1c2/h2o_100ps
```

## Timing report

For `--run` the driver prints:

* wall time (including model load / JIT startup) and LAMMPS loop time,
* ms/step and us/atom/step,
* the 1 ns projection at the chosen `--timestep` (`1000/timestep` steps).

## Programmatic use

```python
from sampling.les_c1c2 import run_md, analyze_trajectory

md = run_md(data=".../six_3000/H2O/data.lammps", species="H2O",
            elements=["C", "H", "N", "O"], temperature=300.0,
            timestep=0.0005, steps=20000, dump_every=1000,
            workdir=".../runs/H2O")
rec, density = analyze_trajectory(md["dump"], outdir=".../c1c2/H2O",
                                  species="H2O",
                                  elements=["C", "H", "N", "O"],
                                  temperature=300.0, pair=("O", "O"))
```
