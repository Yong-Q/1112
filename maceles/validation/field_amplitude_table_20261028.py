#!/usr/bin/env python3
"""Compare external-field energy perturbations with the OFF-POLAR difference.

For the stored 501-frame water trajectory, compute the per-frame field energy
for (a) the four-sine site field at several amplitude scales and seeds and
(b) a uniform electric field acting on the molecular dipoles.  Report the
frame-to-frame std and find the amplitude where it matches the OFF-POLAR
energy-difference std (0.356 eV).
"""
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "validation/off23_500ps_water1152_2026-10-23"
OUT = ROOT / "docs/results/field_amplitude_20261028"
OUT.mkdir(parents=True, exist_ok=True)
KB = 8.617333262e-5
T = 300.0
KT = KB * T
L = 23.3921419057029247
N_ATOMS = 1152
N_MOL = 384
DX = 1.85 * 0.2082  # water dipole in e*A


def read_dump(path, max_frames=200):
    text = Path(path).read_text()
    out = []
    for chunk in text.split("ITEM: TIMESTEP")[1:]:
        body = "ITEM: TIMESTEP" + chunk
        if "ITEM: ATOMS" not in body:
            continue
        head, _, _ = body.partition("ITEM: ATOMS")
        try:
            n = int(head.split("ITEM: NUMBER OF ATOMS")[1].split()[0])
        except Exception:
            continue
        lines = body.split("ITEM: ATOMS", 1)[1].strip().splitlines()[1:]
        if len(lines) < n:
            continue
        arr = np.array([ln.split() for ln in lines[:n]], dtype=float)
        out.append(arr[:, 2:5])
        if len(out) >= max_frames:
            break
    return out


def site_field_energy(pos, scale, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, 4)
    phi = rng.uniform(0.0, 2.0 * np.pi, 4)
    k = 2.0 * np.pi * np.arange(1, 5) / L
    x = pos[:, 0]
    v = np.zeros(len(x))
    for n in range(4):
        v += a[n] * np.sin(k[n] * x + phi[n])
    return KT * scale * v.sum()


def dipole_energy(pos):
    o = pos[0::3]
    h1 = pos[1::3]
    h2 = pos[2::3]
    d = 0.5 * (h1 + h2) - o
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
    return -DX * d[:, 2].sum()  # per V/A


def main():
    frames = read_dump(PKG / "traj_500ps_1fs.lammpstrj", 120)
    print("frames", len(frames), flush=True)
    dU_std = 0.355765  # eV, from the 501-frame CSV
    rows = []
    for seed in (101, 202, 303):
        for scale in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            e = np.array([site_field_energy(p, scale, seed) for p in frames])
            rows.append({"type": "site4", "seed": seed, "amplitude": scale,
                         "std_eV": float(e.std()), "mean_eV": float(e.mean()),
                         "std_meV_atom": float(e.std() / N_ATOMS * 1e3),
                         "ratio_to_model": float(e.std() / dU_std)})
    dip = np.array([dipole_energy(p) for p in frames])
    for E in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        e = dip * E
        rows.append({"type": "uniform", "seed": 0, "amplitude": E,
                     "std_eV": float(e.std()), "mean_eV": float(e.mean()),
                     "std_meV_atom": float(e.std() / N_ATOMS * 1e3),
                     "ratio_to_model": float(e.std() / dU_std)})
    (OUT / "summary.json").write_text(json.dumps(
        {"n_frames": len(frames), "model_std_eV": dU_std, "rows": rows},
        indent=1))
    # markdown table
    lines = ["| type | seed | amplitude | std(ΔV) eV | std meV/atom | "
             "ratio to model ΔU |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        amp = (f"{r['amplitude']:.3f}" if r["type"] == "uniform"
               else f"{r['amplitude']:g}")
        lines.append(f"| {r['type']} | {r['seed']} | {amp} | "
                     f"{r['std_eV']:.4f} | {r['std_meV_atom']:.4f} | "
                     f"{r['ratio_to_model']:.3f} |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
