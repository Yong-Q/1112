#!/usr/bin/env python3
"""Dense scan for the field amplitude where the perturbation matches the model.

std(Delta V) is linear in the field amplitude, so measure the per-unit
amplitude std on all 501 frames and 10 field seeds, then solve for:
  std = 0.3558 eV (model difference),
  sigma = 1, 3 at T = 300/500/700/1000 K.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "validation/off23_500ps_water1152_2026-10-23"
OUT = ROOT / "docs/results/field_amplitude_20261028"
KB = 8.617333262e-5
T0 = 300.0
KT = KB * T0
L = 23.3921419057029247
N = 1152


def read_dump(path):
    text = Path(path).read_text()
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
        yield arr[:, 2:5]


def site_std(pos, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, 4)
    phi = rng.uniform(0.0, 2.0 * np.pi, 4)
    k = 2.0 * np.pi * np.arange(1, 5) / L
    x = pos[:, 0]
    v = np.zeros(len(x))
    for n in range(4):
        v += a[n] * np.sin(k[n] * x + phi[n])
    return KT * scale * v.sum()


def dipole_unit(pos):
    o = pos[0::3]
    h1 = pos[1::3]
    h2 = pos[2::3]
    d = 0.5 * (h1 + h2) - o
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-12
    return -1.85 * 0.2082 * d[:, 2].sum()


def main():
    frames = list(read_dump(PKG / "traj_500ps_1fs.lammpstrj"))
    print("frames", len(frames), flush=True)
    model_std = 0.3558
    # site: std per unit scale for 10 seeds
    s_std = []
    for seed in (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010):
        e = np.array([site_std(p, seed) for p in frames])
        s_std.append(e.std())
    s_std = np.array(s_std)
    # uniform: std per V/A
    e_uni = np.array([dipole_unit(p) for p in frames])
    uni_std = e_uni.std()

    def crossing(target_std):
        return model_std / target_std

    res = {
        "n_frames": len(frames),
        "site_std_per_unit": s_std.tolist(),
        "site_std_mean": float(s_std.mean()),
        "site_std_seed_std": float(s_std.std(ddof=1)),
        "uniform_std_per_V_A": float(uni_std),
        "s_star_model": float(crossing(s_std.mean())),
        "s_star_seed_range": [float(crossing(s_std.max())),
                              float(crossing(s_std.min()))],
        "E_star_model": float(crossing(uni_std)),
        "crossings": {},
    }
    for T in (300, 500, 700, 1000):
        kt = KB * T
        res["crossings"][str(T)] = {
            "site_sigma1": float(kt / s_std.mean()),
            "site_sigma3": float(3 * kt / s_std.mean()),
            "site_model": float(model_std / s_std.mean()),
            "uniform_sigma1": float(kt / uni_std),
            "uniform_sigma3": float(3 * kt / uni_std),
            "uniform_model": float(model_std / uni_std),
        }
    (OUT / "dense_scan.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
