#!/usr/bin/env python3
"""Compare reweighted OFF c2 against direct MACE-POLAR c2 at one state."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "external-repos/graph_electrostatics-0.4.0"))
import ase.io
from mace.calculators import mace_polar
from sampling.direct_correlation import c2_from_g
from sampling.off_polar import OffPolarReweighter

BASE = ROOT / "validation/results/forcefield_c1c2_water96"
OUT = BASE / "reweight_vs_direct"
OUT.mkdir(parents=True, exist_ok=True)
POLAR_MODEL = "/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model"
T = 300.0
BETA = 1.0 / (8.617333262e-5 * T)

off = list(ase.io.iread(BASE / "MACE-OFF_traj.xyz", format="extxyz"))[10:]
pol = list(ase.io.iread(BASE / "MACE-POLAR_traj.xyz", format="extxyz"))[10:]
rho = sum(1 for s in off[0].get_chemical_symbols() if s == "O") / off[0].get_volume()
calc = mace_polar(model=POLAR_MODEL, device="cuda", default_dtype="float64")

dE = []
for i, atoms in enumerate(off):
    atoms.calc = calc
    e = float(atoms.get_potential_energy())
    dE.append(e - float(atoms.info["OFF_energy"]))
dE = np.asarray(dE)
w = np.exp(-(BETA * dE - (BETA * dE).min()))
w /= w.sum()
ess = float(1.0 / np.sum(w ** 2))
sigma = float((BETA * dE).std(ddof=1))

r, g_off = OffPolarReweighter.rdf(off, pair=("O", "O"), rmax=8.0, bins=80)
_, g_rw = OffPolarReweighter.rdf(off, weights=w, pair=("O", "O"), rmax=8.0, bins=80)
_, g_dir = OffPolarReweighter.rdf(pol, pair=("O", "O"), rmax=8.0, bins=80)

r, c2_off, _, _ = c2_from_g(r, g_off, rho)
_, c2_rw, _, _ = c2_from_g(r, g_rw, rho)
_, c2_dir, _, _ = c2_from_g(r, g_dir, rho)

mask = (r > 0.5) & (r < 7.0)
payload = {
    "temperature_K": T, "rho_A3": rho,
    "n_off_frames": len(off), "n_polar_frames": len(pol),
    "sigma_beta": sigma, "ess": ess, "ess_fraction": ess / len(off),
    "c2_direct_vs_off_rms": float(np.sqrt(np.mean((c2_dir[mask] - c2_off[mask]) ** 2))),
    "c2_direct_vs_reweighted_rms": float(np.sqrt(np.mean((c2_dir[mask] - c2_rw[mask]) ** 2))),
    "c2_direct_vs_reweighted_max": float(np.max(np.abs(c2_dir[mask] - c2_rw[mask]))),
    "c2_off_range": [float(c2_off[mask].min()), float(c2_off[mask].max())],
    "c2_reweighted_range": [float(c2_rw[mask].min()), float(c2_rw[mask].max())],
    "c2_direct_range": [float(c2_dir[mask].min()), float(c2_dir[mask].max())],
    "r": r.tolist(), "g_off": g_off.tolist(), "g_reweighted": g_rw.tolist(),
    "g_direct_polar": g_dir.tolist(),
    "c2_off": c2_off.tolist(), "c2_reweighted": c2_rw.tolist(),
    "c2_direct_polar": c2_dir.tolist(),
}
(OUT / "comparison.json").write_text(json.dumps(payload, indent=1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(r, g_off, label="MACE-OFF")
ax[0].plot(r, g_rw, "--", label=f"reweighted (ESS={ess:.1f})")
ax[0].plot(r, g_dir, ":", label="MACE-POLAR direct")
ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("g$_{OO}$(r)")
ax[1].plot(r, c2_off, label="MACE-OFF")
ax[1].plot(r, c2_rw, "--", label="reweighted")
ax[1].plot(r, c2_dir, ":", label="MACE-POLAR direct")
ax[1].set_xlabel("r (Å)"); ax[1].set_ylabel("c$_2$(r)")
for a in ax: a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "reweight_vs_direct.png", dpi=150)
print(json.dumps({k: payload[k] for k in (
    "sigma_beta", "ess", "ess_fraction", "c2_direct_vs_off_rms",
    "c2_direct_vs_reweighted_rms", "c2_direct_vs_reweighted_max")}, indent=1))
