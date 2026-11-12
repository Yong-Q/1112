"""Short NVT trajectory runner used as the source ensemble for reweighting."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import ase
import ase.io
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


@dataclass
class TrajectoryConfig:
    temperature_K: float = 300.0
    timestep_fs: float = 0.5
    ps: float = 5.0
    equil_ps: float = 0.5
    frame_ps: float = 0.1
    friction: float = 0.01
    seed: int = 20261023


def read_structure(path: str | Path) -> ase.Atoms:
    if str(path).endswith(".data"):
        atoms = ase.io.read(path, format="lammps-data", atom_style="atomic")
    else:
        atoms = ase.io.read(path)
    atoms.pbc = True
    return atoms


def run_langevin(atoms: ase.Atoms, calculator: Any, config: TrajectoryConfig,
                 out_path: str | Path,
                 log: Callable[[str], None] = print) -> int:
    """Run NVT Langevin and append frames with source energy/forces."""
    atoms = atoms.copy()
    atoms.calc = calculator
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=config.temperature_K,
        rng=np.random.default_rng(config.seed))
    dyn = Langevin(
        atoms, timestep=config.timestep_fs * units.fs,
        temperature_K=config.temperature_K, friction=config.friction,
        rng=np.random.default_rng(config.seed + 1), fixcm=False)

    steps_per_ps = 1.0 / (config.timestep_fs * 1e-3)
    n_equil = max(0, int(round(config.equil_ps * steps_per_ps)))
    n_total = max(n_equil + 1, int(round(config.ps * steps_per_ps)))
    n_frame = max(1, int(round(config.frame_ps * steps_per_ps)))

    log(f"[traj] atoms={len(atoms)} T={config.temperature_K} K "
        f"equil={n_equil} steps total={n_total} frame_every={n_frame}")
    t0 = time.time()
    if n_equil:
        dyn.run(n_equil)
    log(f"[traj] equilibration done in {time.time() - t0:.1f}s")

    frame = 0
    for step in range(n_equil, n_total + 1, n_frame):
        if step > n_equil:
            dyn.run(n_frame)
        e = float(atoms.get_potential_energy())
        f = np.asarray(atoms.get_forces(), dtype=np.float64)
        if not np.isfinite(e) or not np.isfinite(f).all():
            raise RuntimeError(f"non-finite source result at step {step}")
        snap = atoms.copy()
        snap.calc = None
        snap.info["OFF_energy"] = e
        snap.info["temperature_K"] = config.temperature_K
        snap.info["step"] = step
        snap.arrays["OFF_forces"] = f
        ase.io.write(str(out), snap, format="extxyz", append=True)
        frame += 1
        log(f"[traj] frame {frame:03d} step {step} E={e:.6f} eV "
            f"T_inst={atoms.get_temperature():.1f} K "
            f"wall={time.time() - t0:.1f}s")
    log(f"[traj] wrote {frame} frames to {out}")
    return frame
