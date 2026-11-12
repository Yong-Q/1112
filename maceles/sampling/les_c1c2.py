#!/usr/bin/env python3
"""MACELES/LES-driven c1/c2 extraction for bulk fluids.

This is a thin driver that connects the fixed Symmetrix-MACELES LAMMPS engine
(the build containing the latent-Ewald-summation charge-response force fix)
to the existing cDFT extraction pipeline in :mod:`sampling.cdft_dataset`.
No correlation function is reimplemented here: both modes finish by calling
``cdft_dataset.process_bulk``.

Modes
-----
``--run``
    Write a LAMMPS input for the MACELES-OFF (CHNO) model, run NVT on a
    ``data.lammps`` box with the LES pair style, time the run, and then hand
    the produced dump to ``process_bulk``.  Outputs go to ``<outdir>/f1`` and
    ``<outdir>/f2``; a ``<outdir>/les_c1c2.json`` manifest records the engine,
    input, timing and extraction metadata.

``--analyze-only``
    Skip MD and extract c1/c2 from an existing dump (e.g. the 100 ps
    ``h2o_100ps`` production trajectory) with the same ``process_bulk`` call.

Usage
-----
Short H2O test (3000 atoms, 1000 steps x 0.5 fs = 0.5 ps)::

    python sampling/les_c1c2.py --run \\
        --data /workspace/group/shared/bza/symmetrix-polar/bench/q4_maceles_off/six_3000/H2O/data.lammps \\
        --species H2O --elements "C H N O" \\
        --temperature 300 --timestep 0.0005 \\
        --steps 1000 --dump-every 100 \\
        --c2-pair O O \\
        --outdir /workspace/group/shared/bza/test/sampling/.scratch_maceles/h2o_short \\
        --workdir /workspace/group/shared/bza/test/sampling/.scratch_maceles/run_h2o

The six 3000-atom boxes under
``.../bench/q4_maceles_off/six_3000/<species>/data.lammps`` all use LAMMPS
types C=1 H=2 N=3 O=4, so ``--elements "C H N O"`` is correct for every one
of them.  Species / molecule counts / densities (molecules per A^3) are::

    H2O   1000 molecules   rho=0.0334   --c2-pair O O
    NH3    750 molecules   rho=0.0241   --c2-pair N N
    CH4    600 molecules   rho=0.0158   --c2-pair C C
    C2H4   500 molecules   rho=0.0123   --c2-pair C C
    C2H6   375 molecules   rho=0.0105   --c2-pair C C
    CO2   1000 molecules   rho=0.0137   --c2-pair C O

``--density`` is in molecules/A^3.  If omitted it is derived from the data
file (``--run``) or from the first dump frame (``--analyze-only``), using
``n_atoms / (atoms_per_molecule * volume)``.

Reuse an existing 100 ps production trajectory without rerunning MD::

    python sampling/les_c1c2.py --analyze-only \\
        --traj /workspace/group/shared/bza/symmetrix-polar/bench/q4_maceles_off/h2o_100ps/attempt-01/traj.dump \\
        --species H2O --elements "C H N O" --T 300 --density 0.0334 \\
        --c2-pair O O \\
        --outdir /workspace/group/shared/bza/test/sampling/.scratch_maceles/h2o_100ps_c1c2

Programmatic use::

    from sampling.les_c1c2 import run_md, analyze_trajectory
    md = run_md(data=..., species="H2O", elements=["C", "H", "N", "O"],
                temperature=300.0, timestep=0.0005, steps=20000,
                dump_every=1000, workdir=...)
    rec, density = analyze_trajectory(md["dump"], outdir=..., species="H2O",
                                      elements=["C", "H", "N", "O"],
                                      temperature=300.0, pair=("O", "O"))

Notes
-----
* LAMMPS units are ``metal``: ``--timestep`` is in ps and ``--temperature``
  in K.  The 1 ns estimate uses ``1000 / timestep`` steps.
* The engine must be launched with KOKKOS enabled; this driver uses
  ``lmp -k on g <gpus>``.  The default ``LD_LIBRARY_PATH`` and
  ``OMP_NUM_THREADS=1`` values match the q4 benchmark environment.
* Default model/cubin/binary paths can be overridden with ``--model``,
  ``--cubin`` and ``--lmp``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling.cdft_dataset import process_bulk, read_traj, sha256  # noqa: E402

DEFAULT_LMP = Path(
    "/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp")
DEFAULT_MODEL = Path(
    "/publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/"
    "maceles_off_chno.json")
DEFAULT_CUBIN = Path(
    "/publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/"
    "maceles_off.cubin")
DEFAULT_HOST_ARTIFACT = Path(
    "/publicfs10/fs10-m9/home/m9s005258/.cache/symmetrix/execution-jit/"
    "artifacts/8d00f0566cc8a5f6a603befb0eab51f47d3a025c19c9b9d92e0731eb351e84f7/"
    "factorized_host_plugin_gen11_src0c65a7c611c15feff220748dc034a7a963176ec834be421668b0f9c04a5ca982.so")
DEFAULT_LD_LIBRARY_PATH = (
    "/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:"
    "/publicfs10/fs10-share/soft/share-soft/openmpi/openmpi-4.1.8/install-new/lib:"
    "/publicfs10/fs10-share/soft/share-soft/gcc/gcc-14.2.0/install/lib64")

# GPU path (single process, no MPI) keeps the no_domain_decomposition device
# path.  CPU runs omit the mode keyword so the pair style auto-selects
# no_domain_decomposition (nprocs==1) or mpi_message_passing (nprocs>1), and
# use the host plugin (direct) or the interpretive schedule (generic).
GPU_PAIR_STYLE = (
    "symmetrix/mace/float32/kk no_domain_decomposition streamed_edges "
    "direct profile speed jit_device_artifact {cubin} "
    "jit_device_blocks_per_compute_unit 8")
HOST_DIRECT_PAIR_STYLE = (
    "symmetrix/mace/float32/kk streamed_edges direct profile capacity "
    "jit_host_artifact {host_artifact}")
HOST_GENERIC_PAIR_STYLE = (
    "symmetrix/mace/float32/kk streamed_edges generic profile capacity")


def pair_style_line(cubin=None, host_artifact=None):
    """Return the ``pair_style`` body for the selected backend/edge mode."""
    if host_artifact:
        return HOST_DIRECT_PAIR_STYLE.format(
            host_artifact=Path(host_artifact).resolve())
    if cubin:
        return GPU_PAIR_STYLE.format(cubin=Path(cubin).resolve())
    # No artefact: fall back to the interpretive (generic) CPU schedule.
    return HOST_GENERIC_PAIR_STYLE

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LOOP_RE = re.compile(
    r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs for\s+"
    r"(\d+)\s+steps with\s+(\d+)\s+atoms")
_PERF_RE = re.compile(
    r"Performance:\s*([0-9.eE+-]+)\s*ns/day.*?([0-9.eE+-]+)\s*timesteps/s")
_WALL_RE = re.compile(r"Total wall time:\s*(\d+):(\d+):(\d+)")


def parse_elements(values) -> list[str]:
    """Flatten ``--elements`` values, accepting ``"C H N O"`` or ``C H N O``."""
    out: list[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    if not out:
        raise ValueError("empty --elements")
    return out


def parse_pair(values) -> tuple[str, str]:
    """Parse ``--c2-pair "O O"`` or ``--c2-pair O O`` into a 2-tuple."""
    toks: list[str] = []
    for value in values:
        toks.extend(str(value).replace(",", " ").split())
    if len(toks) != 2:
        raise ValueError(f"--c2-pair needs exactly 2 elements, got {toks}")
    return toks[0], toks[1]


def atoms_per_molecule(species: str) -> int:
    """Number of atoms in a simple formula such as H2O, C2H4, CO2."""
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", str(species))
    if not parts:
        raise ValueError(f"cannot parse species formula {species!r}")
    return sum(int(count) if count else 1 for _, count in parts)


def _species_elements(species: str) -> list[str]:
    return [sym for sym, _ in re.findall(r"([A-Z][a-z]?)(\d*)", str(species))]


def parse_lammps_data(path) -> dict:
    """Read atom count, type count and orthogonal box lengths from a data file."""
    path = Path(path)
    n_atoms = n_types = None
    lengths: dict[str, float] = {}
    rx_atoms = re.compile(r"^\s*(\d+)\s+atoms\s*$")
    rx_types = re.compile(r"^\s*(\d+)\s+atom types\s*$")
    rx_box = re.compile(
        rf"^\s*({_NUM})\s+({_NUM})\s+([xyz])lo\s+[xyz]hi\s*$")
    rx_tilt = re.compile(rf"^\s*({_NUM})\s+({_NUM})\s+({_NUM})\s+xy\s+xz\s+yz\s*$")
    with open(path) as fh:
        for line in fh:
            m = rx_atoms.match(line)
            if m:
                n_atoms = int(m.group(1))
                continue
            m = rx_types.match(line)
            if m:
                n_types = int(m.group(1))
                continue
            m = rx_box.match(line)
            if m:
                lengths[m.group(3)] = float(m.group(2)) - float(m.group(1))
                continue
            m = rx_tilt.match(line)
            if m and any(abs(float(g)) > 0.0 for g in m.groups()):
                raise ValueError(f"{path}: triclinic boxes are not supported")
    missing = [k for k in ("x", "y", "z") if k not in lengths]
    if n_atoms is None or n_types is None or missing:
        raise ValueError(
            f"{path}: could not parse atoms/atom types/box bounds "
            f"(atoms={n_atoms}, types={n_types}, missing={missing})")
    volume = lengths["x"] * lengths["y"] * lengths["z"]
    return {"n_atoms": n_atoms, "n_types": n_types,
            "lengths_A": lengths, "volume_A3": volume}


def _check_composition(elements, species, pair=None) -> None:
    unknown = [s for s in _species_elements(species) if s not in elements]
    if unknown:
        raise ValueError(
            f"species {species!r} contains {unknown}, not in --elements {elements}")
    if pair is not None:
        bad = [s for s in pair if s not in elements]
        if bad:
            raise ValueError(f"--c2-pair {pair} contains {bad}, not in --elements {elements}")


def write_lammps_input(workdir, data, model, cubin, elements, temperature,
                       timestep, steps, dump_every, equil_steps=0,
                       seed=12345, host_artifact=None) -> Path:
    """Write the LES NVT input file and return its path."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    t_start = min(300.0, float(temperature))
    if equil_steps and equil_steps > 0:
        # Ramp from a gentle start temperature to the target over the first
        # 20 % of the equilibration, then hold.  A cold start plus a staged
        # heat-up avoids dumping the full target kinetic energy into a
        # structure that has just been rescaled.
        heat_steps = max(1, int(0.2 * equil_steps))
        equil_block = (
            f"fix 1 all nvt temp {t_start:g} {temperature:g} 0.1\n"
            f"run {heat_steps}\n"
            f"unfix 1\n"
            f"fix 1 all nvt temp {temperature:g} {temperature:g} 0.1\n"
            f"run {equil_steps - heat_steps}\n"
            f"reset_timestep 0\n"
        )
    else:
        equil_block = f"fix 1 all nvt temp {temperature:g} {temperature:g} 0.1\n"
    restart_equil = ("write_restart restart_100ps\n"
                     if (equil_steps and equil_steps > 0) else "")
    text = f"""units metal
atom_style molecular/kk
atom_modify map yes
boundary p p p
package kokkos neigh half newton on
newton on
read_data {Path(data).resolve()}
pair_style {pair_style_line(cubin=cubin, host_artifact=host_artifact)}
pair_coeff * * {Path(model).resolve()} {' '.join(elements)}
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes
timestep {timestep:g}
velocity all create {t_start:g} {seed} mom yes rot no dist gaussian
thermo {dump_every}
thermo_style custom step temp pe
run_style verlet/kk
{equil_block}{restart_equil}dump 1 all custom {dump_every} traj.dump id mol type x y z fx fy fz
dump_modify 1 sort id
run {steps}
write_restart restart_1ns
"""
    path = workdir / "in.les_md"
    path.write_text(text)
    return path


def run_engine(lmp, workdir, input_path, log_path, gpus=1, omp_threads=1,
               np_ranks=1, mpirun=None,
               ld_library_path=DEFAULT_LD_LIBRARY_PATH) -> dict:
    """Run the LAMMPS binary with KOKKOS enabled; return invocation metadata.

    On CPU (``gpus == 0``) the engine is launched under ``mpirun -np np_ranks``
    with one OpenMP thread per rank (``-k on t <omp_threads>``).  On GPU the
    original single-process ``-k on g <gpus>`` path is preserved.
    """
    workdir = Path(workdir)
    if gpus and gpus > 0:
        base = [str(lmp), "-k", "on", "g", str(gpus)]
    else:
        base = [str(lmp), "-k", "on", "t", str(omp_threads)]
    base += ["-in", Path(input_path).name, "-log", Path(log_path).name]
    if gpus and gpus > 0:
        cmd = base
    elif np_ranks > 1:
        cmd = [mpirun or "mpirun", "--allow-run-as-root", "--oversubscribe",
               "-np", str(np_ranks)] + base
    else:
        cmd = base
    env = os.environ.copy()
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        ld_library_path + (os.pathsep + old_ld if old_ld else ""))
    env["OMP_NUM_THREADS"] = str(omp_threads)
    stdout_path = workdir / "stdout.log"
    start = time.monotonic()
    with open(stdout_path, "w") as fh:
        proc = subprocess.run(cmd, cwd=str(workdir), env=env,
                              stdout=fh, stderr=subprocess.STDOUT)
    wall = time.monotonic() - start
    if proc.returncode != 0:
        tail = stdout_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(
            f"LAMMPS failed with exit code {proc.returncode}; see "
            f"{log_path} and {stdout_path}\n--- tail ---\n{tail}")
    return {"command": cmd, "cwd": str(workdir.resolve()),
            "env": {"OMP_NUM_THREADS": str(omp_threads),
                    "LD_LIBRARY_PATH": ld_library_path},
            "stdout": str(stdout_path.resolve()),
            "wall_time_s": wall}


def parse_log_timing(log_path, steps=None) -> dict:
    """Extract loop time and LAMMPS performance from a log file."""
    text = Path(log_path).read_text(errors="replace")
    runs = []
    for m in _LOOP_RE.finditer(text):
        runs.append({"loop_time_s": float(m.group(1)),
                     "n_procs": int(m.group(2)),
                     "n_steps": int(m.group(3)),
                     "n_atoms": int(m.group(4))})
    chosen = None
    if steps is not None:
        chosen = next((r for r in runs if r["n_steps"] == steps), None)
    if chosen is None and runs:
        chosen = runs[-1]
    perf = None
    m = _PERF_RE.search(text)
    if m:
        perf = {"ns_per_day": float(m.group(1)),
                "timesteps_per_s": float(m.group(2))}
    total_wall = None
    m = _WALL_RE.search(text)
    if m:
        total_wall = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                      + int(m.group(3)))
    return {"run": chosen, "performance": perf, "total_wall_time_s": total_wall}


def timing_report(loop_time_s, n_steps, n_atoms, timestep_ps) -> dict:
    """Per-step cost and 1 ns projection for a metal-units timestep."""
    per_step_s = float(loop_time_s) / float(n_steps)
    steps_per_ns = 1000.0 / float(timestep_ps)
    seconds_per_ns = per_step_s * steps_per_ns
    return {
        "loop_time_s": float(loop_time_s),
        "per_step_ms": per_step_s * 1e3,
        "us_per_atom_step": per_step_s / float(n_atoms) * 1e6,
        "steps_per_ns": steps_per_ns,
        "seconds_per_ns": seconds_per_ns,
        "hours_per_ns": seconds_per_ns / 3600.0,
        "days_per_ns": seconds_per_ns / 86400.0,
    }


def run_md(data, species, elements, temperature, timestep, steps, dump_every,
           workdir, equil_steps=200000, seed=12345,
           model=DEFAULT_MODEL, cubin=DEFAULT_CUBIN,
           lmp=DEFAULT_LMP, density=None, gpus=0, omp_threads=1, np_ranks=1,
           mpirun=None, host_artifact=DEFAULT_HOST_ARTIFACT,
           ld_library_path=DEFAULT_LD_LIBRARY_PATH) -> dict:
    """Write input, run LES-MD, and return run metadata plus the dump path."""
    data = Path(data)
    workdir = Path(workdir)
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if dump_every <= 0:
        raise ValueError("--dump-every must be positive")
    elements = list(elements)
    _check_composition(elements, species)
    info = parse_lammps_data(data)
    if info["n_types"] > len(elements):
        raise ValueError(
            f"{data} has {info['n_types']} atom types but only "
            f"{len(elements)} elements were given: {elements}")
    n_per_mol = atoms_per_molecule(species)
    if density is None:
        density = info["n_atoms"] / (n_per_mol * info["volume_A3"])
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = write_lammps_input(
        workdir, data, model, cubin, elements, temperature, timestep,
        steps, dump_every, equil_steps=equil_steps, seed=seed,
        host_artifact=host_artifact if gpus == 0 else None)
    log_path = workdir / "log.les_md"
    invocation = run_engine(lmp, workdir, input_path, log_path, gpus=gpus,
                            omp_threads=omp_threads, np_ranks=np_ranks,
                            mpirun=mpirun,
                            ld_library_path=ld_library_path)
    parsed = parse_log_timing(log_path, steps=steps)
    dump = workdir / "traj.dump"
    if not dump.exists() or dump.stat().st_size == 0:
        raise RuntimeError(f"LAMMPS finished but wrote no dump at {dump}")
    run = parsed["run"]
    if run is None:
        raise RuntimeError(f"could not parse 'Loop time' from {log_path}")
    timing = timing_report(run["loop_time_s"], run["n_steps"], run["n_atoms"],
                           timestep)
    cubin = Path(cubin) if cubin else Path()
    host_artifact = Path(host_artifact) if host_artifact else Path()
    return {
        "mode": "run",
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "workdir": str(workdir.resolve()),
        "input": str(input_path.resolve()),
        "log": str(log_path.resolve()),
        "dump": str(dump.resolve()),
        "dump_sha256": sha256(dump),
        "species": species,
        "elements": elements,
        "temperature_K": float(temperature),
        "timestep_ps": float(timestep),
        "steps": int(steps),
        "equil_steps": int(equil_steps),
        "seed": int(seed),
        "dump_every": int(dump_every),
        "n_atoms": info["n_atoms"],
        "n_atom_types": info["n_types"],
        "box_lengths_A": info["lengths_A"],
        "volume_A3": info["volume_A3"],
        "density_mol_A3": float(density),
        "engine": {
            "lmp": str(Path(lmp).resolve()),
            "lmp_sha256": sha256(lmp),
            "model": str(Path(model).resolve()),
            "model_sha256": sha256(model),
            "cubin": str(cubin.resolve()) if cubin else "",
            "cubin_sha256": sha256(cubin) if cubin else "",
            "host_artifact": str(host_artifact.resolve())
                if host_artifact else "",
            "np_ranks": int(np_ranks),
            "gpus": int(gpus),
        },
        "invocation": invocation,
        "timing": timing,
        "lammps_performance": parsed["performance"],
        "lammps_total_wall_time_s": parsed["total_wall_time_s"],
    }


def analyze_trajectory(traj, outdir, species, elements, temperature,
                       density=None, pair=("O", "O"), rmax=None,
                       c2_bins=80, bins=2000, stride=1,
                       fmt="lammps-dump-text") -> tuple[dict, float]:
    """Extract c1/c2 from an existing dump via ``cdft_dataset.process_bulk``."""
    import ase.io

    traj = Path(traj)
    outdir = Path(outdir)
    if not traj.exists():
        raise FileNotFoundError(traj)
    elements = list(elements)
    pair = tuple(pair) if pair is not None else ("O", "O")
    _check_composition(elements, species, pair)
    if density is None:
        if fmt == "lammps-dump-text":
            from ase import Atoms
            from sampling.dump_mol import read_dump
            f = next(iter(read_dump(traj, elements=elements, max_frames=1)))
            first = Atoms(symbols=f["symbols"], positions=f["positions"],
                          cell=f["cell"], pbc=True)
        else:
            first = next(ase.io.iread(str(traj), format=fmt))
        density = len(first) / (atoms_per_molecule(species)
                                * first.get_volume())
    outdir.mkdir(parents=True, exist_ok=True)
    rec = process_bulk(traj, outdir, elements, species, float(temperature),
                       float(density), fmt=fmt, stride=stride, pair=pair,
                       rmax=rmax, c2_bins=c2_bins, bins=bins)
    if rec is None:
        raise RuntimeError(f"no frames could be read from {traj}")
    return rec, float(density)


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _write_manifest(outdir, payload) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "les_c1c2.json"
    path.write_text(json.dumps(_jsonable(payload), indent=1) + "\n")
    return path


def _print_summary(outdir, rec, density) -> None:
    print(f"[les_c1c2] F1/F2 written under {Path(outdir).resolve()}")
    print(f"[les_c1c2] F1: n_frames={rec['n_frames']} "
          f"rho={density:.6g} molecules/A^3 "
          f"({rec['n_frames']} frames, flat rho + zero Vext)")
    print(f"[les_c1c2] F2: c2 pair={tuple(rec['c2_pair'])} "
          f"rho_pair={rec['rho_pair_A3']:.6g} A^-3 "
          f"rmax={rec['c2_rmax_A']:.3g} A bins={rec['c2_bins']} "
          f"short_trajectory={rec['c2_short_trajectory']}")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="MACELES/LES MD -> c1/c2 via sampling.cdft_dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true",
                      help="run LES-MD from --data, then extract c1/c2")
    mode.add_argument("--analyze-only", action="store_true",
                      help="skip MD and extract c1/c2 from --traj")
    ap.add_argument("--data", default=None,
                    help="LAMMPS data file for --run")
    ap.add_argument("--traj", default=None,
                    help="existing LAMMPS dump for --analyze-only")
    ap.add_argument("--traj-format", default="lammps-dump-text",
                    choices=["lammps-dump-text", "extxyz"],
                    help="trajectory format for --analyze-only")
    ap.add_argument("--species", required=True,
                    help="molecule formula, e.g. H2O, NH3, CH4, C2H4, C2H6, CO2")
    ap.add_argument("--elements", nargs="+", default=["C", "H", "N", "O"],
                    help='type->element order, e.g. "C H N O" (types C=1 H=2 N=3 O=4)')
    ap.add_argument("--temperature", "--T", dest="temperature", type=float,
                    default=300.0, help="NVT temperature in K")
    ap.add_argument("--timestep", type=float, default=0.0005,
                    help="MD timestep in ps (metal units)")
    ap.add_argument("--steps", type=int, default=2000000,
                    help="number of MD steps for --run")
    ap.add_argument("--dump-every", type=int, default=1000,
                    help="dump (and thermo) interval in steps")
    ap.add_argument("--equil-steps", type=int, default=200000,
                    help="NVT equilibration steps before production/dump "
                         "(default 200000 = 100 ps at 0.5 fs; 0 disables)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="velocity initialization seed")
    ap.add_argument("--outdir", required=True,
                    help="output directory for f1/, f2/ and les_c1c2.json")
    ap.add_argument("--workdir", default=None,
                    help="LAMMPS run directory (default: <outdir>/run)")
    ap.add_argument("--density", type=float, default=None,
                    help="molecules/A^3; derived from data/dump if omitted")
    ap.add_argument("--c2-pair", nargs="+", default=["O", "O"],
                    metavar="EL",
                    help='element pair for the c2 RDF, "O O" or O O '
                         '(default O O for water)')
    ap.add_argument("--rmax", type=float, default=None,
                    help="c2 RDF cutoff in A (default: min(8, L/2))")
    ap.add_argument("--bins", type=int, default=2000,
                    help="F1 density-profile bins")
    ap.add_argument("--c2-bins", type=int, default=80,
                    help="c2 RDF bins")
    ap.add_argument("--stride", type=int, default=1,
                    help="read every Nth frame when extracting c1/c2")
    ap.add_argument("--lmp", default=str(DEFAULT_LMP),
                    help="LAMMPS binary")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="MACELES model JSON")
    ap.add_argument("--cubin", default=str(DEFAULT_CUBIN),
                    help="JIT device artifact for the model (GPU only)")
    ap.add_argument("--host-artifact", default=str(DEFAULT_HOST_ARTIFACT),
                    help="JIT host plugin for the model (CPU direct)")
    ap.add_argument("--gpus", type=int, default=0,
                    help="GPUs per rank for KOKKOS (-k on g N); 0 = CPU")
    ap.add_argument("--np-ranks", type=int, default=1,
                    help="MPI ranks for CPU runs (mpirun -np N)")
    ap.add_argument("--mpirun", default=None,
                    help="mpirun binary (default: mpirun on PATH)")
    ap.add_argument("--omp-threads", default="1",
                    help="OMP_NUM_THREADS for the engine")
    ap.add_argument("--ld-library-path", default=DEFAULT_LD_LIBRARY_PATH,
                    help="LD_LIBRARY_PATH for the engine")
    return ap


def main(argv=None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    elements = parse_elements(args.elements)
    try:
        pair = parse_pair(args.c2_pair) if args.c2_pair else ("O", "O")
    except ValueError as exc:
        ap.error(str(exc))
    outdir = Path(args.outdir)

    if args.run:
        if not args.data:
            ap.error("--run requires --data")
        workdir = Path(args.workdir) if args.workdir else outdir / "run"
        md = run_md(
            data=args.data, species=args.species, elements=elements,
            temperature=args.temperature, timestep=args.timestep,
            steps=args.steps, dump_every=args.dump_every, workdir=workdir,
            equil_steps=args.equil_steps, seed=args.seed,
            model=args.model, cubin=args.cubin, lmp=args.lmp,
            density=args.density, gpus=args.gpus,
            omp_threads=args.omp_threads, np_ranks=args.np_ranks,
            mpirun=args.mpirun, host_artifact=args.host_artifact,
            ld_library_path=args.ld_library_path)
        t = md["timing"]
        print(f"[les_c1c2] engine={md['engine']['lmp']}")
        print(f"[les_c1c2] input={md['input']}")
        print(f"[les_c1c2] log={md['log']}")
        print(f"[les_c1c2] wall={md['invocation']['wall_time_s']:.2f} s "
              f"(incl. startup/JIT), loop={t['loop_time_s']:.3f} s for "
              f"{md['steps']} steps x {md['n_atoms']} atoms")
        print(f"[les_c1c2] per-step={t['per_step_ms']:.3f} ms  "
              f"{t['us_per_atom_step']:.3f} us/atom/step")
        print(f"[les_c1c2] 1 ns @ {md['timestep_ps']:g} ps = "
              f"{t['steps_per_ns']:.4g} steps -> "
              f"{t['hours_per_ns']:.4g} h ({t['days_per_ns']:.4g} days)")
        if md["lammps_performance"]:
            p = md["lammps_performance"]
            print(f"[les_c1c2] LAMMPS: {p['ns_per_day']:.4g} ns/day, "
                  f"{p['timesteps_per_s']:.4g} timesteps/s")
        rec, density = analyze_trajectory(
            md["dump"], outdir, args.species, elements, args.temperature,
            density=md["density_mol_A3"], pair=pair, rmax=args.rmax,
            c2_bins=args.c2_bins, bins=args.bins, stride=args.stride)
        payload = {
            "schema": "les-c1c2-v1",
            "generated_by": "sampling.les_c1c2",
            "mode": "run",
            "elements": elements,
            "c2_pair": list(pair),
            "md": md,
            "c1c2": rec,
            "outputs": {
                "f1_dir": str((outdir / "f1").resolve()),
                "f2_dir": str((outdir / "f2").resolve()),
            },
        }
    else:
        if not args.traj:
            ap.error("--analyze-only requires --traj")
        rec, density = analyze_trajectory(
            args.traj, outdir, args.species, elements, args.temperature,
            density=args.density, pair=pair, rmax=args.rmax,
            c2_bins=args.c2_bins, bins=args.bins, stride=args.stride,
            fmt=args.traj_format)
        payload = {
            "schema": "les-c1c2-v1",
            "generated_by": "sampling.les_c1c2",
            "mode": "analyze-only",
            "elements": elements,
            "c2_pair": list(pair),
            "traj": str(Path(args.traj).resolve()),
            "traj_sha256": sha256(args.traj),
            "c1c2": rec,
            "outputs": {
                "f1_dir": str((outdir / "f1").resolve()),
                "f2_dir": str((outdir / "f2").resolve()),
            },
        }

    manifest = _write_manifest(outdir, payload)
    _print_summary(outdir, rec, density)
    print(f"[les_c1c2] manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
