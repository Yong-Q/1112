"""Interface for reweighting a source MLIP ensemble to a target MLIP.

The core identity is free-energy perturbation / importance sampling:

    p_B(x) = p_A(x) exp(-beta [V_B(x) - V_A(x)]) Z_A / Z_B
    <O>_B = <O exp(-beta dV)>_A / <exp(-beta dV)>_A

A constant energy offset cancels in the normalized weights.  The practical
question is the overlap: sigma_beta = std(beta dV) and the effective sample
size ESS = (sum w)^2 / sum w^2.  See
docs/design/OFF_TO_POLAR_REWEIGHTING_2026-10-23.md for the derivation and
citations.

The API is deliberately calculator-agnostic: pass any pair of ASE
calculators, or use `from_models` for the MACE-OFF / MACE-POLAR pair.

Example
-------
    from sampling.off_polar import OffPolarReweighter, ReweightConfig
    rw = OffPolarReweighter.from_models(
        source_model="MACE-OFF23_medium.model",
        target_model="MACE-POLAR-1-M.model",
        config=ReweightConfig(temperature_K=300.0),
    )
    result = rw.evaluate(frames)
    result.write("outdir")
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GRAPH_LONG_RANGE = Path(os.environ.get(
    "GRAPH_LONG_RANGE_PATH",
    str(REPO.parent / "external-repos/graph_electrostatics-0.4.0")))

KB_EV = 8.617333262e-5


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class ReweightConfig:
    temperature_K: float = 300.0
    stride: int = 1
    max_frames: int = 0
    device: str = "cuda"
    dtype: str = "float64"
    source_dtype: str = "float32"
    rdf_pair: tuple[str, str] = ("O", "O")
    rdf_rmax_A: float = 8.0
    rdf_bins: int = 80

    @property
    def beta_eV_inv(self) -> float:
        return 1.0 / (KB_EV * self.temperature_K)


@dataclass
class FrameRecord:
    index: int
    n_atoms: int
    e_source_eV: float
    e_target_eV: float
    dE_total_eV: float
    dE_meV_per_atom: float
    beta_dE: float
    weight: float
    force_mae_meV_per_A: float = float("nan")


@dataclass
class ReweightResult:
    config: ReweightConfig
    source_model: str = ""
    target_model: str = ""
    source_sha256: str = ""
    target_sha256: str = ""
    frames: list[FrameRecord] = field(default_factory=list)
    rdf_source: tuple[np.ndarray, np.ndarray] | None = None
    rdf_weighted: tuple[np.ndarray, np.ndarray] | None = None

    # aggregate properties -------------------------------------------------
    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_atoms(self) -> int:
        return self.frames[0].n_atoms if self.frames else 0

    @property
    def dE_meV_per_atom(self) -> np.ndarray:
        return np.array([f.dE_meV_per_atom for f in self.frames])

    @property
    def weights(self) -> np.ndarray:
        return np.array([f.weight for f in self.frames])

    @property
    def beta_dE(self) -> np.ndarray:
        return np.array([f.beta_dE for f in self.frames])

    @property
    def sigma_beta(self) -> float:
        return float(self.beta_dE.std(ddof=1)) if self.n_frames > 1 else 0.0

    @property
    def ess(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w ** 2)) if len(w) else 0.0

    @property
    def ess_fraction(self) -> float:
        return self.ess / self.n_frames if self.n_frames else 0.0

    def _linear_fit(self) -> tuple[float, float, float, float]:
        eo = np.array([f.e_source_eV for f in self.frames])
        ep = np.array([f.e_target_eV for f in self.frames])
        x = eo - eo.mean()
        y = ep - ep.mean()
        c1 = float(np.dot(x, y) / np.dot(x, x))
        c2 = float(ep.mean() - c1 * eo.mean())
        resid = y - c1 * x
        ss_res = float(np.dot(resid, resid))
        ss_tot = float(np.dot(y, y))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        resid_atom = float(np.sqrt(np.mean((resid / self.n_atoms) ** 2)) * 1000.0)
        return c1, c2, r2, resid_atom

    @property
    def linear_slope(self) -> float:
        return self._linear_fit()[0]

    @property
    def linear_intercept_eV(self) -> float:
        return self._linear_fit()[1]

    @property
    def linear_r2(self) -> float:
        return self._linear_fit()[2]

    @property
    def linear_residual_rms_meV_per_atom(self) -> float:
        return self._linear_fit()[3]

    @property
    def verdict(self) -> str:
        if self.ess_fraction >= 0.10 and self.sigma_beta <= 3.0:
            return "usable"
        if self.sigma_beta <= 5.0:
            return "marginal"
        return "collapse"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "off_to_polar_reweight",
            "source_model": self.source_model,
            "target_model": self.target_model,
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "temperature_K": self.config.temperature_K,
            "beta_eV_inv": self.config.beta_eV_inv,
            "n_frames": self.n_frames,
            "n_atoms": self.n_atoms,
            "constant_offset_meV_per_atom": float(self.dE_meV_per_atom.mean()),
            "delta_meV_per_atom_std": float(self.dE_meV_per_atom.std(ddof=1))
            if self.n_frames > 1 else 0.0,
            "sigma_beta": self.sigma_beta,
            "ess": self.ess,
            "ess_fraction": self.ess_fraction,
            "linear": {
                "slope": self.linear_slope,
                "intercept_eV": self.linear_intercept_eV,
                "r2": self.linear_r2,
                "residual_rms_meV_per_atom":
                    self.linear_residual_rms_meV_per_atom,
            },
            "force_mae_meV_per_A_mean": float(
                np.nanmean([f.force_mae_meV_per_A for f in self.frames])),
            "verdict": self.verdict,
            "frames": [asdict(f) for f in self.frames],
        }

    def write(self, outdir: str | Path) -> Path:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.as_dict()
        (out / "reweight_manifest.json").write_text(json.dumps(summary, indent=1))
        with open(out / "reweight_frames.csv", "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame", "n_atoms", "dE_total_eV",
                             "dE_meV_per_atom", "beta_dE", "weight",
                             "force_mae_meV_per_A"])
            for f in self.frames:
                writer.writerow([f.index, f.n_atoms, f.e_target_eV - f.e_source_eV,
                                 f.dE_meV_per_atom, f.beta_dE, f.weight,
                                 f.force_mae_meV_per_A])
        self._plot(out / "reweight_diagnostics.png")
        return out / "reweight_manifest.json"

    def reweight_observable(self, values: Sequence[float]) -> float:
        vals = np.asarray(values, dtype=float)
        if len(vals) != self.n_frames:
            raise ValueError("observable length must match frames")
        w = self.weights
        return float(np.sum(w * vals) / np.sum(w))

    def _plot(self, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idx = np.arange(self.n_frames)
        d = self.dE_meV_per_atom
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        ax[0, 0].plot(idx, d, "o-", ms=3, lw=0.8)
        ax[0, 0].set_xlabel("frame")
        ax[0, 0].set_ylabel("ΔE (meV/atom)")
        ax[0, 0].set_title("target − source")
        ax[0, 1].hist(d, bins=min(20, max(5, self.n_frames // 2)),
                      color="C0", alpha=0.75)
        ax[0, 1].set_xlabel("ΔE (meV/atom)")
        ax[0, 1].set_ylabel("frames")
        ax[0, 1].set_title(f"σ(βΔE)={self.sigma_beta:.2f}, "
                           f"ESS={self.ess:.1f}")
        eo = np.array([f.e_source_eV for f in self.frames]) / self.n_atoms
        ep = np.array([f.e_target_eV for f in self.frames]) / self.n_atoms
        ax[1, 0].plot(eo, ep, "o", ms=4)
        ax[1, 0].set_xlabel("source E (eV/atom)")
        ax[1, 0].set_ylabel("target E (eV/atom)")
        ax[1, 0].set_title(f"slope={self.linear_slope:.5f}, R²={self.linear_r2:.4f}")
        if self.rdf_source is not None and self.rdf_weighted is not None:
            ax[1, 1].plot(*self.rdf_source, label="source")
            ax[1, 1].plot(*self.rdf_weighted, "--",
                          label=f"reweighted (ESS={self.ess:.1f})")
            ax[1, 1].set_xlabel("r (Å)")
            ax[1, 1].set_ylabel("g(r)")
            ax[1, 1].set_title("pair RDF")
            ax[1, 1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


class OffPolarReweighter:
    """Evaluate a source trajectory with a target calculator and reweight."""

    def __init__(self, source_calc: Any, target_calc: Any,
                 config: ReweightConfig | None = None):
        self.source_calc = source_calc
        self.target_calc = target_calc
        self.config = config or ReweightConfig()

    @classmethod
    def from_models(cls, source_model: str, target_model: str,
                    config: ReweightConfig | None = None) -> "OffPolarReweighter":
        """Build the MACE-OFF source and MACE-POLAR target calculators."""
        cfg = config or ReweightConfig()
        if str(GRAPH_LONG_RANGE) not in sys.path:
            sys.path.insert(0, str(GRAPH_LONG_RANGE))
        from mace.calculators import MACECalculator, mace_polar
        source = MACECalculator(model_paths=source_model, device=cfg.device,
                                default_dtype=cfg.source_dtype)
        target = mace_polar(model=target_model, device=cfg.device,
                            default_dtype=cfg.dtype)
        obj = cls(source, target, cfg)
        obj.source_model = str(Path(source_model).resolve())
        obj.target_model = str(Path(target_model).resolve())
        obj.source_sha256 = sha256_file(source_model)
        obj.target_sha256 = sha256_file(target_model)
        return obj

    def evaluate(self, frames: Iterable[Any],
                 source_energy_key: str = "OFF_energy",
                 source_forces_key: str = "OFF_forces") -> ReweightResult:
        """Evaluate frames, compute ΔE, weights, ESS and linear diagnostics."""
        cfg = self.config
        frames = list(frames)[:: cfg.stride]
        if cfg.max_frames:
            frames = frames[: cfg.max_frames]
        if len(frames) < 2:
            raise ValueError("need at least 2 frames")

        out = ReweightResult(config=cfg,
                             source_model=getattr(self, "source_model", ""),
                             target_model=getattr(self, "target_model", ""),
                             source_sha256=getattr(self, "source_sha256", ""),
                             target_sha256=getattr(self, "target_sha256", ""))
        e_src, e_tgt, d, fmae, natoms = [], [], [], [], []
        for i, atoms in enumerate(frames):
            n = len(atoms)
            if source_energy_key in atoms.info:
                es = float(atoms.info[source_energy_key])
            else:
                atoms.calc = self.source_calc
                es = float(atoms.get_potential_energy())
            fs = atoms.arrays.get(source_forces_key)
            atoms.calc = self.target_calc
            et = float(atoms.get_potential_energy())
            ft = np.asarray(atoms.get_forces(), dtype=np.float64)
            if not np.isfinite(et) or not np.isfinite(ft).all():
                raise RuntimeError(f"non-finite target result on frame {i}")
            fm = float("nan")
            if fs is not None:
                fs = np.asarray(fs, dtype=np.float64)
                if fs.shape == ft.shape:
                    fm = float(np.mean(np.abs(ft - fs)))
            e_src.append(es)
            e_tgt.append(et)
            d.append(et - es)
            fmae.append(fm)
            natoms.append(n)

        e_src = np.asarray(e_src)
        e_tgt = np.asarray(e_tgt)
        d = np.asarray(d)
        natoms = np.asarray(natoms)
        beta_d = cfg.beta_eV_inv * d
        w_raw = np.exp(-(beta_d - beta_d.min()))
        w = w_raw / w_raw.sum()
        for i in range(len(frames)):
            out.frames.append(FrameRecord(
                index=i, n_atoms=int(natoms[i]),
                e_source_eV=float(e_src[i]), e_target_eV=float(e_tgt[i]),
                dE_total_eV=float(d[i]),
                dE_meV_per_atom=float(d[i] / natoms[i] * 1000.0),
                beta_dE=float(beta_d[i]), weight=float(w[i]),
                force_mae_meV_per_A=float(fmae[i] * 1000.0),
            ))
        out.rdf_source = self.rdf(frames, pair=cfg.rdf_pair,
                                  rmax=cfg.rdf_rmax_A, bins=cfg.rdf_bins)
        out.rdf_weighted = self.rdf(frames, weights=w, pair=cfg.rdf_pair,
                                    rmax=cfg.rdf_rmax_A, bins=cfg.rdf_bins)
        return out

    @staticmethod
    def rdf(frames: Sequence[Any], weights: Sequence[float] | None = None,
            pair: tuple[str, str] = ("O", "O"), rmax: float = 8.0,
            bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(0.0, rmax, bins + 1)
        hist = np.zeros(bins)
        norm = 0.0
        for k, atoms in enumerate(frames):
            w = 1.0 if weights is None else float(weights[k])
            if w == 0.0:
                continue
            syms = np.asarray(atoms.get_chemical_symbols())
            ia = np.where(syms == pair[0])[0]
            ib = np.where(syms == pair[1])[0]
            dist = atoms.get_all_distances(mic=True)
            block = dist[np.ix_(ia, ib)]
            if pair[0] == pair[1]:
                iu = np.triu_indices(len(ia), k=1)
                r = block[iu]
                expected = 0.5 * len(ia) * (len(ia) - 1)
            else:
                r = block.reshape(-1)
                expected = len(ia) * len(ib)
            h, _ = np.histogram(r, bins=edges)
            hist += w * h
            vol = atoms.get_volume()
            # Spherical-shell volume 4*pi/3*(r_{i+1}^3 - r_i^3).  Using the
            # squared-radius difference here would under-count every shell by
            # ~3r/2 and inflate g(r) by the same factor.
            shell = 4.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
            norm += w * expected * shell / vol
        g = hist / np.maximum(norm, 1e-30)
        mid = 0.5 * (edges[1:] + edges[:-1])
        return mid, g
