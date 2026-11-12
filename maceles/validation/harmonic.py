#!/usr/bin/env python3
"""Analytic validation of the reweighting estimator.

Two exact checks:

1. 1D harmonic: source N(0, 1/(beta k)), target N(x0, 1/(beta k)).
   The reweighted mean and second moment must converge to x0 and
   x0^2 + 1/(beta k).
2. N independent modes shifted by x0: sigma_beta = x0 sqrt(beta k N), so the
   direct exponential average collapses as sqrt(N).  This is the scaling
   discussed in Majumdar et al. 2026, SI Sec. 8.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 8.617333262e-5
T = 300.0
BETA = 1.0 / (KB * T)
K = 5.5e-18 / 1.602176634e-19  # 5.5e-18 J/A^2 -> eV/A^2


def weights_from_dU(dU):
    b = BETA * np.asarray(dU, dtype=float)
    w = np.exp(-(b - b.min()))
    return w / w.sum()


def ess(w):
    return float(1.0 / np.sum(w ** 2))


def one_dimensional(n=20000, x0=0.02, seed=1):
    rng = np.random.default_rng(seed)
    sigma = 1.0 / np.sqrt(BETA * K)
    x = rng.normal(0.0, sigma, n)
    dU = 0.5 * K * ((x - x0) ** 2 - x ** 2)
    w = weights_from_dU(dU)
    mean_x = float(np.sum(w * x))
    mean_x2 = float(np.sum(w * x ** 2))
    exact_mean = x0
    exact_x2 = x0 ** 2 + sigma ** 2
    return {
        "n": n, "x0_A": x0, "sigma_A": sigma,
        "exact_mean": exact_mean, "reweighted_mean": mean_x,
        "mean_error": mean_x - exact_mean,
        "exact_x2": exact_x2, "reweighted_x2": mean_x2,
        "x2_error": mean_x2 - exact_x2,
        "sigma_beta": float((BETA * dU).std(ddof=1)),
        "ess": ess(w), "ess_fraction": ess(w) / n,
    }


def n_modes(n_modes, n_samples=4000, x0=0.02, seed=2):
    rng = np.random.default_rng(seed)
    sigma = 1.0 / np.sqrt(BETA * K)
    x = rng.normal(0.0, sigma, (n_samples, n_modes))
    # dU = sum_i [0.5 k (x_i-x0)^2 - 0.5 k x_i^2]
    dU = np.sum(0.5 * K * ((x - x0) ** 2 - x ** 2), axis=1)
    w = weights_from_dU(dU)
    return {
        "n_modes": int(n_modes),
        "n_samples": int(n_samples),
        "x0_A": x0,
        "sigma_beta_measured": float((BETA * dU).std(ddof=1)),
        "sigma_beta_theory": float(x0 * np.sqrt(BETA * K * n_modes)),
        "ess": ess(w),
        "ess_fraction": ess(w) / n_samples,
    }


def main() -> int:
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)
    conv = [one_dimensional(n=n, seed=10 + i)
            for i, n in enumerate([100, 1000, 10000, 100000])]
    scaling = [n_modes(n) for n in [1, 4, 16, 64, 256, 1024]]
    payload = {"temperature_K": T, "k_eV_A2": K,
               "convergence": conv, "scaling": scaling}
    (out / "harmonic_validation.json").write_text(json.dumps(payload, indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    n = np.array([c["n"] for c in conv])
    ax[0].loglog(n, np.abs([c["mean_error"] for c in conv]), "o-",
                 label="|mean error|")
    ax[0].loglog(n, np.abs([c["x2_error"] for c in conv]), "s-",
                 label="|x² error|")
    ax[0].set_xlabel("source samples")
    ax[0].set_ylabel("absolute error (Å or Å²)")
    ax[0].set_title("1D exact harmonic")
    ax[0].legend()
    m = np.array([s["n_modes"] for s in scaling])
    sb = np.array([s["sigma_beta_measured"] for s in scaling])
    st = np.array([s["sigma_beta_theory"] for s in scaling])
    ax[1].loglog(m, sb, "o-", label="measured σ(βΔU)")
    ax[1].loglog(m, st, "--", label="x₀√(βkN)")
    ax[1].axhline(3.0, color="C3", ls=":", label="σ=3")
    ax[1].set_xlabel("number of shifted modes N")
    ax[1].set_ylabel("σ(βΔU)")
    ax[1].set_title("collapse scales as √N")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "harmonic_validation.png", dpi=150)
    plt.close(fig)

    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
