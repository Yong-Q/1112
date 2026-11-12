#!/usr/bin/env python3
"""Plot the 501-frame MACE-OFF23 and MACE-POLAR potential-energy fluctuations."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "validation/off23_500ps_water1152_2026-10-23"
OUT = PKG / "potential_fluctuations_501"

rows = []
with open(PKG / "frames_501_total_energies.csv") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)
t = np.array([float(r["time_ps"]) for r in rows])
n = int(rows[0]["n_atoms"])
e_off = np.array([float(r["E_OFF_total_eV"]) for r in rows])
e_pol = np.array([float(r["E_POLAR_total_eV"]) for r in rows])
dU = e_pol - e_off

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "savefig.dpi": 300,
})
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#4D4D4D"

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
a = ax[0]
a.plot(t, (e_off - e_off.mean()) / n * 1e3, lw=0.8, color=BLUE,
       label="MACE-OFF23")
a.plot(t, (e_pol - e_pol.mean()) / n * 1e3, lw=0.8, color=ORANGE,
       label="MACE-POLAR")
a.set_xlabel("time (ps)")
a.set_ylabel("(E - <E>)/N (meV atom$^{-1}$)")
a.set_title("Potential-energy fluctuations")
a.legend(frameon=False)
a.grid(alpha=0.2)

b = ax[1]
b.plot(t, dU, lw=0.7, color=GREY, label="$E_{POLAR}-E_{OFF}$")
b.axhline(dU.mean(), color=BLUE, ls="--", lw=0.9,
          label=f"mean {dU.mean():.2f} eV")
b.axhspan(dU.mean() - dU.std(), dU.mean() + dU.std(), color=BLUE,
          alpha=0.12, label=f"$\\pm1\\sigma$ = {dU.std():.3f} eV")
b.set_xlabel("time (ps)")
b.set_ylabel("$\\Delta U$ (eV, whole box)")
b.set_title("Target-minus-source energy difference")
b.legend(frameon=False, fontsize=7.5)
b.grid(alpha=0.2)

for a_ in ax:
    a_.spines["top"].set_visible(False)
    a_.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
print("wrote", OUT.with_suffix(".png"))
print(f"mean dU {dU.mean():.4f} eV  std {dU.std():.4f} eV  "
      f"per atom std {dU.std()/n*1e3:.4f} meV/atom")


# ------------------------------------------------------------------ combined
fig2, ax2 = plt.subplots(1, 2, figsize=(10.5, 3.8))
a = ax2[0]
a.plot(t, (e_off - e_off.mean()) / n * 1e3, lw=0.7, color=BLUE,
       label="MACE-OFF23")
a.plot(t, (e_pol - e_pol.mean()) / n * 1e3, lw=0.7, color=ORANGE,
       label="MACE-POLAR")
a.plot(t, (dU - dU.mean()) / n * 1e3, lw=1.0, color="#009E73",
       label="$\\Delta U$ (POLAR $-$ OFF23)")
a.set_xlabel("time (ps)")
a.set_ylabel("(E - <E>)/N (meV atom$^{-1}$)")
a.set_title("Thermal fluctuations vs energy difference")
a.legend(frameon=False, fontsize=7.5)
a.grid(alpha=0.2)

b = ax2[1]
vals = [e_off.std() / n * 1e3, e_pol.std() / n * 1e3, dU.std() / n * 1e3]
cols = [BLUE, ORANGE, "#009E73"]
bars = b.bar(["E_OFF", "E_POLAR", "$\\Delta U$"], vals, color=cols,
             width=0.6)
for bar, v in zip(bars, vals):
    b.text(bar.get_x() + bar.get_width() / 2, v * 1.15, f"{v:.3f}",
           ha="center", fontsize=8)
b.set_ylabel("std (meV atom$^{-1}$)")
b.set_yscale("log")
b.set_title("Fluctuation magnitude")
b.set_ylim(top=max(vals) * 3.0)
b.text(0.97, 0.06,
       f"whole-box $\\beta\\Delta U$: $\\sigma$ = {dU.std()/(8.617333262e-5*300):.2f}",
       transform=b.transAxes, fontsize=8, va="bottom", ha="right")
b.grid(alpha=0.2, axis="y")
for a_ in ax2:
    a_.spines["top"].set_visible(False)
    a_.spines["right"].set_visible(False)
fig2.tight_layout()
fig2.savefig(OUT.with_name(OUT.name + "_combined.png"), bbox_inches="tight")
fig2.savefig(OUT.with_name(OUT.name + "_combined.pdf"), bbox_inches="tight")
print("wrote combined figure")


# ------------------------------------------------------------------ single panel
fig3, a3 = plt.subplots(figsize=(9.0, 4.2))
a3.plot(t, (e_off - e_off.mean()) / n * 1e3, lw=0.7, color=BLUE,
        label="MACE-OFF23")
a3.plot(t, (e_pol - e_pol.mean()) / n * 1e3, lw=0.7, color=ORANGE,
        label="MACE-POLAR")
a3.plot(t, (dU - dU.mean()) / n * 1e3, lw=1.1, color="#009E73",
        label="$\\Delta U$ (POLAR $-$ OFF23)")
a3.set_xlabel("time (ps)")
a3.set_ylabel("(E - <E>)/N (meV atom$^{-1}$)")
a3.set_title("Potential-energy fluctuations of the 501-frame trajectory")
a3.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
a3.grid(alpha=0.2)
a3.spines["top"].set_visible(False)
a3.spines["right"].set_visible(False)
a3.text(0.01, 0.03,
        f"std: E_OFF {e_off.std()/n*1e3:.2f}, E_POLAR {e_pol.std()/n*1e3:.2f}, "
        f"$\\Delta U$ {dU.std()/n*1e3:.2f} meV/atom",
        transform=a3.transAxes, fontsize=7.5, va="bottom")
fig3.tight_layout()
fig3.savefig(OUT.with_name(OUT.name + "_single.png"), bbox_inches="tight")
fig3.savefig(OUT.with_name(OUT.name + "_single.pdf"), bbox_inches="tight")
print("wrote single-panel figure")


# ------------------------------------------------------------------ bath view
import re
log = PKG / "log.production_1fs"
step_t, temp_t = [], []
if log.exists():
    for line in log.read_text().splitlines():
        m = re.match(r"^\s*(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s*$", line)
        if m:
            step_t.append(int(m.group(1)) * 0.001)
            temp_t.append(float(m.group(2)))
step_t = np.array(step_t)
temp_t = np.array(temp_t)

fig4, (a4, b4) = plt.subplots(2, 1, figsize=(9.0, 5.0), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.2]})
a4.plot(t, (e_off - e_off.mean()) / n * 1e3, lw=0.7, color=BLUE,
        label="MACE-OFF23")
a4.plot(t, (e_pol - e_pol.mean()) / n * 1e3, lw=0.7, color=ORANGE,
        label="MACE-POLAR")
a4.plot(t, (dU - dU.mean()) / n * 1e3, lw=1.1, color="#009E73",
        label="$\\Delta U$ (POLAR $-$ OFF23)")
for val, col in [(e_off.std() / n * 1e3, BLUE),
                 (dU.std() / n * 1e3, "#009E73")]:
    a4.axhline(val, color=col, ls=":", lw=0.8)
    a4.axhline(-val, color=col, ls=":", lw=0.8)
a4.fill_between(t, -e_off.std() / n * 1e3, e_off.std() / n * 1e3,
                color=BLUE, alpha=0.07)
a4.set_ylabel("(E - <E>)/N (meV atom$^{-1}$)")
a4.set_title("Potential-energy fluctuations and the heat-bath envelope")
a4.legend(frameon=False, ncol=3, fontsize=7.5, loc="upper right")
a4.grid(alpha=0.2)
a4.text(0.01, 0.03,
        "dotted: $\\pm1\\sigma$ of E_OFF (blue) and $\\Delta U$ (green); "
        "shaded: $\\pm1\\sigma$ thermal envelope",
        transform=a4.transAxes, fontsize=7.2, va="bottom")
if len(step_t):
    b4.plot(step_t, temp_t, lw=0.7, color="#CC79A7")
    b4.axhline(temp_t.mean(), color=GREY, ls="--", lw=0.8)
    b4.set_ylabel("T (K)")
    b4.set_xlabel("time (ps)")
    b4.set_title(f"Thermostat temperature (mean {temp_t.mean():.1f} K, "
                 f"std {temp_t.std():.1f} K)", fontsize=9)
    b4.grid(alpha=0.2)
    b4.spines["top"].set_visible(False)
    b4.spines["right"].set_visible(False)
a4.spines["top"].set_visible(False)
a4.spines["right"].set_visible(False)
fig4.tight_layout()
fig4.savefig(OUT.with_name(OUT.name + "_bath.png"), bbox_inches="tight")
fig4.savefig(OUT.with_name(OUT.name + "_bath.pdf"), bbox_inches="tight")
print("wrote bath figure")
