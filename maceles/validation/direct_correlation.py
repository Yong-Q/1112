#!/usr/bin/env python3
"""Numerical validation of the Ornstein-Zernike c2 inversion.

1. Smooth analytic c(r): construct h(k)=c(k)/(1-rho c(k)), inverse to h(r),
   g=1+h, then recover c2(r) with c2_from_g.  Should match to ~1e-12.
2. Percus-Yevick hard spheres: use the analytic PY c(r) to build g(r), then
   recover c2.  The discontinuous c(r) at contact produces Gibbs ringing at
   r=0; compare the smooth part r>0.2.
3. Ideal gas: g=1 -> c2=0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sampling.direct_correlation import c2_from_g


def transform(r, f):
    dr = r[1] - r[0]
    L = 2.0 * (r[-1] + dr)
    kmax = np.pi / dr
    dk = 2.0 * np.pi / L
    k = np.arange(0.0, kmax + 0.5 * dk, dk)
    kr = np.outer(k, r)
    sinc = np.ones_like(kr)
    np.divide(np.sin(kr), kr, out=sinc, where=kr != 0)
    return k, 4.0 * np.pi * dr * ((r ** 2 * f) @ sinc.T), sinc, dk


def inverse(k, fk, r, sinc, dk):
    return (dk / (2.0 * np.pi ** 2)) * ((k ** 2 * fk) @ sinc)


def main() -> int:
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    payload = {}

    # smooth analytic c
    rho, sig, c0 = 0.2, 0.5, -5.0
    r = np.arange(0.0, 15.0, 0.01)
    c = c0 * np.exp(-0.5 * (r / sig) ** 2)
    k, ck, sinc, dk = transform(r, c)
    h = inverse(k, ck / (1.0 - rho * ck), r, sinc, dk)
    _, c2, _, _ = c2_from_g(r, 1.0 + h, rho)
    payload["smooth_gaussian"] = {
        "rho": rho, "c0": c0, "sigma": sig,
        "max_abs_error": float(np.max(np.abs(c2 - c))),
        "c2_at_zero": float(c2[0]), "c_exact_at_zero": float(c[0]),
    }

    # PY hard spheres
    eta, d = 0.4, 1.0
    rho = 6.0 * eta / (np.pi * d ** 3)
    r = np.arange(0.0, 20.0, 0.01)
    x = r / d
    c = np.zeros_like(r)
    m = x < 1.0
    c[m] = -((1 + 2 * eta) ** 2
             - 6 * eta * (1 + eta / 2) ** 2 * x[m]
             + 0.5 * eta * (1 + 2 * eta) ** 2 * x[m] ** 3) / (1 - eta) ** 4
    k, ck, sinc, dk = transform(r, c)
    h = inverse(k, ck / (1.0 - rho * ck), r, sinc, dk)
    _, c2, _, _ = c2_from_g(r, 1.0 + h, rho)
    mask = (r > 0.2) & (r < 0.9)
    payload["py_hard_sphere"] = {
        "eta": eta, "rho": rho,
        "max_abs_error_0.2_to_0.9": float(np.max(np.abs(c2[mask] - c[mask]))),
        "c2_at_zero": float(c2[0]), "c_exact_at_zero": float(c[0]),
        "note": "discontinuous c(r) causes Gibbs ringing at r=0",
    }

    # ideal gas
    r = np.arange(0.05, 8.0, 0.05)
    _, c2, _, _ = c2_from_g(r, np.ones_like(r), 0.02)
    payload["ideal_gas"] = {"max_abs_c2": float(np.max(np.abs(c2)))}

    (out / "direct_correlation_validation.json").write_text(
        json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
