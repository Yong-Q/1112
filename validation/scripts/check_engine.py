#!/usr/bin/env python3
"""Compare engine energies/forces against the PyTorch reference in a validation npz.

    python check_engine.py H2O_50frames_engine_vs_pytorch.npz
"""
import sys, json
import numpy as np

LIMITS = {"dE_rms_eV": 1e-2, "dF_mae_eV_A": 1e-3, "F_corr_min": 0.9999}


def main(path):
    d = np.load(path)
    Ee, Er = d["energy_engine"].astype(float), d["energy_pytorch"].astype(float)
    Fe, Fr = d["forces_engine"].astype(float), d["forces_pytorch"].astype(float)
    dE, dF = Er - Ee, Fr - Fe
    res = dict(frames=int(len(Ee)), atoms=int(Fe.shape[1]),
               dE_max_eV=float(np.abs(dE).max()), dE_rms_eV=float(np.sqrt((dE ** 2).mean())),
               dE_per_atom_meV=float(1e3 * np.abs(dE).mean() / Fe.shape[1]),
               dF_max_eV_A=float(np.abs(dF).max()), dF_mae_eV_A=float(np.abs(dF).mean()),
               dF_rms_eV_A=float(np.sqrt((dF ** 2).mean())),
               F_corr=float(np.corrcoef(Fe.ravel(), Fr.ravel())[0, 1]))
    print(json.dumps(res, indent=1))
    ok = (res["dE_rms_eV"] < LIMITS["dE_rms_eV"] and
          res["dF_mae_eV_A"] < LIMITS["dF_mae_eV_A"] and
          res["F_corr"] > LIMITS["F_corr_min"])
    print("PASS" if ok else "FAIL", "limits:", LIMITS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
