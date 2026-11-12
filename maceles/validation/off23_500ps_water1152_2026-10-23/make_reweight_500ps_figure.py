#!/usr/bin/env python3
"""Plots for the 500 ps / 1152-atom OFF->POLAR reweighting diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REW = HERE.parent / "bench/off23_500ps_water1152_2026-10-23/reweight"


def main() -> int:
    frames = json.loads((REW / "frames.json").read_text())
    summary = json.loads((REW / "summary.json").read_text())
    rdf = json.loads((REW / "rdf.json").read_text())

    ts = np.array([f["timestep"] for f in frames]) / 1000.0  # 1 fs steps -> ps
    dpa = np.array([f["dE_total_eV"] / f["n_atoms"] for f in frames]) * 1e3
    beta = 1.0 / (8.617333262e-5 * summary["temperature_K"])
    bd = beta * np.array([f["dE_total_eV"] for f in frames])
    lw = -(bd - bd.min())
    w = np.exp(lw)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(ts, dpa, lw=0.8, color="tab:blue")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"$\Delta E$ (meV/atom)")
    ax.set_title(f"OFF$\\to$POLAR energy gap, 1152 H$_2$O, 500 ps\n"
                 f"mean {dpa.mean():.3f}, std {dpa.std():.3f} meV/atom")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.hist(bd - bd.mean(), bins=40, color="tab:purple", alpha=0.8)
    ax.set_xlabel(r"$\beta\Delta E - \langle\beta\Delta E\rangle$")
    ax.set_ylabel("frames")
    ax.set_title(fr"$\sigma_\beta$ = {summary['sigma_beta']:.2f}, "
                 f"ESS = {ess:.2f}/{len(frames)} = {100*ess/len(frames):.2f}%")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    order = np.argsort(w)[::-1]
    ax.semilogy(np.arange(1, len(w) + 1), w[order] / w.max(), "o-", ms=2, lw=0.8)
    ax.set_xlabel("frame rank")
    ax.set_ylabel(r"$w / w_{\max}$")
    ax.set_title("weight spectrum (collapse is a single-frame dominance)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    r = np.array(rdf["r"])
    ax.plot(r, rdf["g_off"], label="OFF (unweighted)", lw=1.4)
    ax.plot(r, rdf["g_reweighted"], label="OFF $\\to$ POLAR reweighted",
            lw=1.4, alpha=0.85)
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel(r"$r_{OO}$ ($\AA$)")
    ax.set_ylabel(r"$g_{OO}(r)$")
    ax.set_title("O–O RDF: reweighting adds noise, no new structure")
    ax.set_xlim(2.2, 8.0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("MACE-OFF $\\to$ MACE-POLAR reweighting on a 500 ps trajectory "
                 "(1152 atoms, 300 K)", fontsize=12)
    fig.tight_layout()
    out = HERE / "figures" / "OFF_TO_POLAR_500PS_1152.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
