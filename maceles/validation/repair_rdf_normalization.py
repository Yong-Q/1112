#!/usr/bin/env python3
"""Repair stored RDF/c2 results produced before the shell-volume fix.

The original ``OffPolarReweighter.rdf`` used

    shell_bad = 4*pi/3 * (r_hi^2 - r_lo^2)

instead of the spherical-shell volume

    shell_good = 4*pi/3 * (r_hi^3 - r_lo^3).

The histogram itself was correct, so every stored g(r) is an exact linear
image of the true one:

    g_true(r) = g_bad(r) * shell_bad / shell_good
              = g_bad(r) * (r_hi^2-r_lo^2) / (r_hi^3-r_lo^3).

This script applies that correction to the JSON products that were generated
from the 96-atom trajectories, truncates the transform at r <= L/2 (5.75 A;
beyond that the minimum-image convention folds the RDF), recomputes c2 with
the Ornstein-Zernike inversion, and rewrites the comparison metrics.

The trajectories themselves are not stored in this repository; the correction
is exact because it acts on the stored g arrays, not on a refit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sampling.direct_correlation import c2_from_g  # noqa: E402

BASE = ROOT / "validation/results/forcefield_c1c2_water96"
R_MAX = 5.75  # L = 11.696 A; L/2 = 5.848 A


def correction(r: np.ndarray) -> np.ndarray:
    dr = float(r[1] - r[0])
    lo = r - 0.5 * dr
    hi = r + 0.5 * dr
    return (hi ** 2 - lo ** 2) / (hi ** 3 - lo ** 3)


def corrected_g(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(payload["r_A"], dtype=float)
    g = np.asarray(payload["g_OO"], dtype=float) * correction(r)
    m = r <= R_MAX
    return r[m], g[m]


def rewrite_c2_file(path: Path, rho: float) -> dict:
    old = json.loads(path.read_text())
    r, g = corrected_g(old)
    r, c2, k, c2k = c2_from_g(r, g, rho)
    payload = dict(old)
    payload.update({
        "r_A": r.tolist(), "g_OO": g.tolist(),
        "c2_r": c2.tolist(), "k_A_inv": k.tolist(), "c2_k": c2k.tolist(),
        "g_min": float(g.min()), "g_max": float(g.max()),
        "c2_min": float(c2[(r > 0.8)].min()),
        "c2_max": float(c2[(r > 0.8)].max()),
        "rmax_c2_A": R_MAX,
        "rdf_normalization": "4*pi/3*(r_hi^3-r_lo^3); corrected 2026-10-23",
    })
    path.write_text(json.dumps(payload, indent=1))
    return payload


def main() -> int:
    rho = 0.02
    off = rewrite_c2_file(BASE / "MACE-OFF_c2.json", rho)
    pol = rewrite_c2_file(BASE / "MACE-POLAR_c2.json", rho)

    # verification.json only stores summary ranges; c1 comes from Widom and is
    # unaffected by the RDF normalization.
    ver_path = BASE / "verification.json"
    ver = json.loads(ver_path.read_text())
    for name, payload in (("MACE-OFF", off), ("MACE-POLAR", pol)):
        s = ver["systems"][name]
        s["g_min"] = payload["g_min"]
        s["g_max"] = payload["g_max"]
        s["c2_min"] = payload["c2_min"]
        s["c2_max"] = payload["c2_max"]
        s["rmax_c2_A"] = R_MAX
    ver["rdf_normalization"] = (
        "corrected 2026-10-23: shell volume 4*pi/3*(r_hi^3-r_lo^3)")
    ver_path.write_text(json.dumps(ver, indent=1))

    comp_path = BASE / "reweight_vs_direct/comparison.json"
    comp = json.loads(comp_path.read_text())
    r = np.asarray(comp["r"], dtype=float)
    fac = correction(r)
    m = r <= R_MAX
    rr = r[m]
    c2 = {}
    for key in ("g_off", "g_reweighted", "g_direct_polar"):
        g = (np.asarray(comp[key], dtype=float) * fac)[m]
        _, c2[key], _, _ = c2_from_g(rr, g, rho)
    sel = (rr > 0.8) & (rr < R_MAX)
    comp.update({
        "r": rr.tolist(),
        "g_off": ((np.asarray(comp["g_off"]) * fac)[m]).tolist(),
        "g_reweighted": ((np.asarray(comp["g_reweighted"]) * fac)[m]).tolist(),
        "g_direct_polar": ((np.asarray(comp["g_direct_polar"]) * fac)[m]).tolist(),
        "c2_off": c2["g_off"].tolist(),
        "c2_reweighted": c2["g_reweighted"].tolist(),
        "c2_direct_polar": c2["g_direct_polar"].tolist(),
        "rmax_c2_A": R_MAX,
        "rdf_normalization": "4*pi/3*(r_hi^3-r_lo^3); corrected 2026-10-23",
    })
    for label, key in (("off", "g_off"), ("reweighted", "g_reweighted"),
                       ("direct", "g_direct_polar")):
        comp[f"c2_{label}_range"] = [float(c2[key][sel].min()),
                                     float(c2[key][sel].max())]
    d_off = c2["g_direct_polar"][sel] - c2["g_off"][sel]
    d_rw = c2["g_direct_polar"][sel] - c2["g_reweighted"][sel]
    comp["c2_direct_vs_off_rms"] = float(np.sqrt(np.mean(d_off ** 2)))
    comp["c2_direct_vs_reweighted_rms"] = float(np.sqrt(np.mean(d_rw ** 2)))
    comp["c2_direct_vs_reweighted_max"] = float(np.max(np.abs(d_rw)))
    comp_path.write_text(json.dumps(comp, indent=1))

    # Refresh the diagnostic figure with the corrected, untruncated-axis data.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(rr, comp["g_off"], label="MACE-OFF")
    ax[0].plot(rr, comp["g_reweighted"], "--", label="reweighted")
    ax[0].plot(rr, comp["g_direct_polar"], ":", label="MACE-POLAR direct")
    ax[0].set_xlim(2.0, R_MAX)
    ax[0].set_xlabel("r (A)"); ax[0].set_ylabel("g$_{OO}$(r)")
    ax[0].legend(fontsize=8)
    ax[1].plot(rr, comp["c2_off"], label="MACE-OFF")
    ax[1].plot(rr, comp["c2_reweighted"], "--", label="reweighted")
    ax[1].plot(rr, comp["c2_direct_polar"], ":", label="MACE-POLAR direct")
    ax[1].set_xlim(0.8, R_MAX)
    ax[1].set_xlabel("r (A)"); ax[1].set_ylabel("c$_2$(r)")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(comp_path.parent / "reweight_vs_direct.png", dpi=150)
    plt.close(fig)

    # Refresh the example c2 product (36-frame example trajectory).
    ex_path = ROOT / "examples/water96/c2/c2.json"
    if ex_path.exists():
        ex = rewrite_c2_file(ex_path, rho)
        print("example c2 range",
              round(ex["c2_min"], 2), round(ex["c2_max"], 2))

    print(json.dumps({
        "MACE-OFF_c2_range": [off["c2_min"], off["c2_max"]],
        "MACE-POLAR_c2_range": [pol["c2_min"], pol["c2_max"]],
        "direct_vs_off_rms": comp["c2_direct_vs_off_rms"],
        "direct_vs_reweighted_rms": comp["c2_direct_vs_reweighted_rms"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
