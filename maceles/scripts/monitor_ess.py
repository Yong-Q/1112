#!/usr/bin/env python3
"""Live ESS monitor for a running LAMMPS trajectory.

Reads complete frames from a LAMMPS dump, evaluates the source and target
potentials on each new frame, and writes running sigma_beta / ESS diagnostics.
No target MD is needed: ESS depends only on the frame weights.

Example:
  python scripts/monitor_ess.py --dump runs/traj.lammpstrj \
    --off-model MACE-OFF23_medium.model --polar-model MACE-POLAR-1-M.model \
    --temperature 300 --outdir runs/ess_live
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import ase.io

KB = 8.617333262e-5


def complete_frames(text: str):
    """Yield (timestep, atoms) for complete LAMMPS dump frames."""
    chunks = text.split("ITEM: TIMESTEP")
    for chunk in chunks[1:]:
        body = "ITEM: TIMESTEP" + chunk
        if "ITEM: ATOMS" not in body:
            continue
        head, _, _ = body.partition("ITEM: ATOMS")
        # number of atoms
        try:
            n_atoms = int(head.split("ITEM: NUMBER OF ATOMS")[1].split()[0])
        except Exception:
            continue
        ts = int(chunk.strip().splitlines()[0])
        atom_lines = body.split("ITEM: ATOMS", 1)[1].strip().splitlines()
        if len(atom_lines) < n_atoms:
            continue
        try:
            atoms = ase.io.read(io.StringIO(body), format="lammps-dump-text")
        except Exception:
            continue
        z = np.asarray(atoms.numbers)
        atoms.set_chemical_symbols(["O" if int(t) == 1 else "H" for t in z])
        yield ts, atoms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--off-model", required=True)
    ap.add_argument("--polar-model", required=True)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
    from mace.calculators import MACECalculator, mace_polar

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    last_ts = -1
    if state_path.exists():
        last_ts = int(json.loads(state_path.read_text()).get("last_timestep", -1))
    rows = []
    csv_path = out / "ess_live.csv"
    if csv_path.exists():
        for line in csv_path.read_text().splitlines()[1:]:
            if line.strip():
                p = line.split(",")
                rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))

    off = MACECalculator(model_paths=args.off_model, device=args.device,
                         default_dtype="float32")
    pol = mace_polar(model=args.polar_model, device=args.device,
                     default_dtype="float64")
    beta = 1.0 / (KB * args.temperature)
    dump = Path(args.dump)

    print(f"[ess] watching {dump}; last_timestep={last_ts}", flush=True)
    while True:
        if dump.exists():
            text = dump.read_text(errors="ignore")
            for ts, atoms in complete_frames(text):
                if ts <= last_ts:
                    continue
                atoms.calc = off
                eo = float(atoms.get_potential_energy())
                atoms.calc = pol
                ep = float(atoms.get_potential_energy())
                d = ep - eo
                last_ts = ts
                rows.append((ts, float(d / len(atoms) * 1000.0), float(beta * d), 0.0))
                if len(rows) >= 2:
                    b = np.array([r[2] for r in rows])
                    w = np.exp(-(b - b.min()))
                    w /= w.sum()
                    ess = float(1.0 / np.sum(w ** 2))
                    sigma = float(b.std(ddof=1))
                    # Store the normalized running weights, not the 0.0
                    # placeholder written when the row was appended.
                    for i, wi in enumerate(w):
                        rows[i] = (rows[i][0], rows[i][1], rows[i][2], float(wi))
                else:
                    ess, sigma = 1.0, 0.0
                    rows[0] = (rows[0][0], rows[0][1], rows[0][2], 1.0)
                print(f"[ess] step={ts} dE/atom={d/len(atoms)*1000:.3f} meV "
                      f"sigma_beta={sigma:.2f} ESS={ess:.2f}/{len(rows)} "
                      f"({100*ess/len(rows):.1f}%)", flush=True)
                with open(csv_path, "w") as fh:
                    fh.write("step,dE_meV_per_atom,beta_dE,weight\n")
                    for r in rows:
                        fh.write(f"{r[0]},{r[1]:.8f},{r[2]:.8f},{r[3]:.8f}\n")
                state_path.write_text(json.dumps({"last_timestep": last_ts,
                                                  "n_frames": len(rows)}))
                (out / "ess_live.json").write_text(json.dumps({
                    "n_frames": len(rows), "last_timestep": last_ts,
                    "sigma_beta": sigma, "ess": ess,
                    "ess_fraction": ess / len(rows),
                    "verdict": ("usable" if sigma <= 3 and ess / len(rows) >= 0.1
                                else "marginal" if sigma <= 5 else "collapse"),
                }, indent=1))
                if args.max_frames and len(rows) >= args.max_frames:
                    return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
