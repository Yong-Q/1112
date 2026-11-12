#!/usr/bin/env python3
"""GPU verification: short MACE-OFF and MACE-POLAR trajectories at one state,
then g(r), c2(r) via OZ and c1 via Widom for both force fields.

This is a qualitative 96-atom/10 ps verification, not a production c2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "external-repos/graph_electrostatics-0.4.0"))

import ase.io
from ase.build import molecule
from sampling.direct_correlation import c1_from_widom, c2_from_g
from sampling.off_polar import OffPolarReweighter
from sampling.trajectory import TrajectoryConfig, read_structure, run_langevin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", required=True)
    ap.add_argument("--off-model", required=True)
    ap.add_argument("--polar-model", required=True)
    ap.add_argument("--student-model", default=None,
                    help="optional distilled MACE-Polar surrogate to compare")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--ps", type=float, default=10.0)
    ap.add_argument("--equil-ps", type=float, default=1.0)
    ap.add_argument("--frame-ps", type=float, default=0.1)
    ap.add_argument("--n-widom", type=int, default=100)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    atoms = read_structure(args.structure)
    rho = sum(1 for s in atoms.get_chemical_symbols() if s == "O") / atoms.get_volume()

    from mace.calculators import MACECalculator, mace_polar
    calcs = {
        "MACE-OFF": MACECalculator(model_paths=args.off_model, device="cuda",
                                   default_dtype="float32"),
        "MACE-POLAR": mace_polar(model=args.polar_model, device="cuda",
                                 default_dtype="float64"),
    }
    if args.student_model:
        calcs["MACE-STUDENT"] = MACECalculator(
            model_paths=args.student_model, device="cuda",
            default_dtype="float32")
    payload = {"temperature_K": args.temperature, "rho_A3": rho,
               "n_atoms": len(atoms), "ps": args.ps,
               "equil_ps": args.equil_ps, "frame_ps": args.frame_ps,
               "systems": {}}
    curves = {}
    for label, calc in calcs.items():
        traj = out / f"{label}_traj.xyz"
        print(f"[gpu] === {label} {args.ps} ps ===", flush=True)
        run_langevin(atoms, calc, TrajectoryConfig(
            temperature_K=args.temperature, ps=args.ps,
            equil_ps=args.equil_ps, frame_ps=args.frame_ps,
            friction=0.05, seed=20261023), traj)
        frames = list(ase.io.iread(traj, format="extxyz"))
        # discard the first 10 frames (~1 ps) again if needed
        frames = frames[10:]
        r, g = OffPolarReweighter.rdf(frames, pair=("O", "O"),
                                      rmax=args.rmax, bins=80)
        r, c2r, k, c2k = c2_from_g(r, g, rho)
        c1 = c1_from_widom(frames[::5][:20], calc, molecule("H2O"),
                           args.temperature, n_insertions=args.n_widom,
                           seed=11)
        curves[label] = (r, g, c2r)
        payload["systems"][label] = {
            "n_frames": len(frames),
            "g_min": float(g.min()), "g_max": float(g.max()),
            "c2_min": float(c2r.min()), "c2_max": float(c2r.max()),
            "c1": c1,
            "r_A": r.tolist(), "g_OO": g.tolist(), "c2_r": c2r.tolist(),
            "k_A_inv": k.tolist(), "c2_k": c2k.tolist(),
        }
        (out / f"{label}_c2.json").write_text(json.dumps(
            payload["systems"][label], indent=1))
        print(f"[gpu] {label}: c1={c1['c1_ex']:.4f} "
              f"c2 min/max={c2r.min():.3f}/{c2r.max():.3f}", flush=True)

    # Pairwise differences against the target (POLAR) on the shared grid.
    if "MACE-POLAR" in payload["systems"]:
        ref = payload["systems"]["MACE-POLAR"]
        gref = np.asarray(ref["g_OO"]); c2ref = np.asarray(ref["c2_r"])
        payload["vs_polar"] = {}
        for label, sysd in payload["systems"].items():
            if label == "MACE-POLAR":
                continue
            g = np.asarray(sysd["g_OO"]); c2 = np.asarray(sysd["c2_r"])
            payload["vs_polar"][label] = {
                "g_rms": float(np.sqrt(np.mean((g - gref) ** 2))),
                "g_max_abs": float(np.max(np.abs(g - gref))),
                "c2_rms": float(np.sqrt(np.mean((c2 - c2ref) ** 2))),
                "c2_max_abs": float(np.max(np.abs(c2 - c2ref))),
                "c1_delta": float(sysd["c1"]["c1_ex"] - ref["c1"]["c1_ex"]),
            }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for label, (r, g, c2r) in curves.items():
        ax[0].plot(r, g, label=label)
        ax[1].plot(r, c2r, label=label)
    ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("g$_{OO}$(r)")
    ax[1].set_xlabel("r (Å)"); ax[1].set_ylabel("c$_2$(r)")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "g_and_c2.png", dpi=150)
    (out / "verification.json").write_text(json.dumps(payload, indent=1))
    print(f"[gpu] wrote {out/'verification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
