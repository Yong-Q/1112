#!/usr/bin/env python3
"""Build PRL/PNAS-style cDFT training data from site-field MD runs.

Two schemas are written from the same trajectory:

PRL (Dijkman et al., PRL 134, 056103) — per simulation i:
    F1 (c1):  rho_i.npy, Vext_i.npy, mu_i.npy      (+ target dF/drho = mu - Vext - ln rho)
    F2 (c2):  rhob_i.npy, c2_i.npy, r_values_i.npy, mu_i.npy, dF_drho_i.npy
PNAS (Bui & Cox, PNAS 2026) — one dict:
    pnas_sim_dict.npy = {sim_id: {"rho": ..., "Vext": ...}}

The external potential is the four-sine field written by
`run_site_symm.make_field`:

    V(x)/eV = sum_n A_n sin(k_n x + phi_n),   k_n = 2 pi n / L

so the 1D profile is taken along x.  Density can be molecular (centre of
mass, default) or atomic.  For molecular profiles Vext is evaluated at the
centre of mass; the intra-molecular variation of the field is recorded as an
approximation in the manifest.

The chemical potential is not known for a fixed-N run.  Default convention:
mu = <ln rho + Vext> over the profile, which makes dF/drho zero-mean and
fixes c1 up to an additive constant (recorded as `mu_convention`).  Pass
--mu to override with a Widom/TI value.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling.direct_correlation import c2_from_g  # noqa: E402

KB_EV_PER_K = 8.617333262e-5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_field(work: Path) -> dict:
    return json.loads((work / "field.json").read_text())


def vext_of_x(x, field: dict) -> np.ndarray:
    """V(x) in eV from the four-sine field."""
    A = np.asarray(field["A_eV"], dtype=float)
    k = np.asarray(field["k_invA"], dtype=float)
    phi = np.asarray(field["phi"], dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.zeros_like(x)
    for a, kk, p in zip(A, k, phi):
        v += a * np.sin(kk * x + p)
    return v


def read_traj(path: Path, elements, stride: int = 1, fmt: str = "lammps-dump-text"):
    import ase.io
    from ase import Atoms
    if fmt == "lammps-dump-text":
        # ASE's dump reader re-scans the whole gzip stream per frame and is
        # orders of magnitude slower than the numpy reader for our dumps.
        from sampling.dump_mol import read_dump
        return [Atoms(symbols=f["symbols"], positions=f["positions"],
                      cell=f["cell"], pbc=True)
                for f in read_dump(path, elements=list(elements),
                                   stride=max(1, stride))]
    it = ase.io.iread(path, format=fmt)
    return [a for i, a in enumerate(it) if i % stride == 0]


def molecule_indices(atoms, scale: float = 1.25):
    """Connected components from a covalent-radius bond criterion."""
    from ase.data import covalent_radii
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from ase.neighborlist import neighbor_list

    r = np.array([covalent_radii[z] for z in atoms.numbers])
    cut = scale * (r[:, None] + r[None, :])
    i, j = neighbor_list("ij", atoms, cut.max())
    d = atoms.get_distances(i, j, mic=True)
    keep = d < cut[i, j]
    n = len(atoms)
    adj = coo_matrix((np.ones(keep.sum()), (i[keep], j[keep])), shape=(n, n))
    ncomp, labels = connected_components(adj, directed=False)
    return labels, ncomp


def com_positions(atoms, labels, ncomp):
    """Molecular centres of mass, minimum-image safe, wrapped into the cell."""
    cell = np.asarray(atoms.cell)
    inv = np.linalg.inv(cell)
    pos = atoms.get_positions()
    masses = atoms.get_masses()
    com = np.zeros((ncomp, 3))
    for c in range(ncomp):
        idx = np.where(labels == c)[0]
        ref = pos[idx[0]]
        frac = (pos[idx] - ref) @ inv
        frac -= np.round(frac)
        delta = frac @ cell
        com[c] = ref + (delta * masses[idx, None]).sum(0) / masses[idx].sum()
    frac = com @ inv
    frac -= np.floor(frac)
    return frac @ cell


def density_profile(frames, field, kind="molecular", bins=2000,
                    weights=None):
    """Return (x_centers, rho_profile, rho_bulk_mean, n_frames).

    With weights w_i (normalised importance weights) the returned profile is
    the target-ensemble estimate sum_i w_i rho_i(x); without weights it is
    the plain source-ensemble average.
    """
    L = float(field["L_A"])
    edges = np.linspace(0.0, L, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    hist = np.zeros(bins)
    wsum = 0.0
    labels = ncomp = None
    for i, atoms in enumerate(frames):
        w = 1.0 if weights is None else float(weights[i])
        if w == 0.0:
            continue
        if kind == "molecular":
            if labels is None:
                labels, ncomp = molecule_indices(atoms)
            pos = com_positions(atoms, labels, ncomp)
        else:
            pos = atoms.get_positions()
        h, _ = np.histogram(pos[:, 0] % L, bins=edges)
        hist += w * h
        wsum += w
    vol = frames[0].get_volume()
    rho = hist / max(wsum, 1e-30) / (vol / bins)
    rho_mean = (hist.sum() / max(wsum, 1e-30)) / vol
    return centers, rho, float(rho_mean), len(frames)


def write_prl_f1(outdir: Path, sim_id: str, rho, vext, mu, cutoff=1e-6):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"rho_{sim_id}.npy", rho.astype(np.float32))
    np.save(outdir / f"Vext_{sim_id}.npy", vext.astype(np.float32))
    np.save(outdir / f"mu_{sim_id}.npy", np.float32(mu))


def write_prl_f2(outdir: Path, sim_id: str, rhob, r, c2r, mu):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / f"rhob_{sim_id}.npy", np.float32(rhob))
    np.save(outdir / f"r_values_{sim_id}.npy", r.astype(np.float32))
    np.save(outdir / f"c2_{sim_id}.npy", c2r.astype(np.float32))
    np.save(outdir / f"mu_{sim_id}.npy", np.float32(mu))
    # bulk dF/drho target: ln(rhob) - c1_bulk.  With the profile-mean mu
    # convention the absolute value is fixed by the reference state; store 0
    # and let the trainer subtract its own bulk reference.
    np.save(outdir / f"dF_drho_{sim_id}.npy", np.float32(0.0))


def process_run(work: Path, outdir: Path, bins=2000, kind="molecular",
                mu_override=None, rmax=None, c2_bins=80, pair=None,
                pnas_dict=None, stride=1, weights=None, c2_on_field=False,
                weights_source="none"):
    field = read_field(work)
    elements = field.get("pair_elements") or ["O", "H"]
    traj = work / "traj.lammpstrj"
    if not traj.exists():
        return None
    frames = read_traj(traj, elements, stride=stride)
    if not frames:
        return None
    sim_id = field.get("tag") or work.name
    species = field.get("species")
    if not species:
        # older 1152-atom runs omit species; recover it from the tag
        stem = sim_id.replace("site-", "").replace("student-site-", "")
        species = stem.split("-")[0] if "-" in stem else "fluid"
    if weights is not None:
        weights = np.asarray(weights, dtype=float)[::stride]
        if len(weights) != len(frames):
            raise ValueError(f"weights length {len(weights)} != frames {len(frames)}")
        weights = weights / weights.sum()

    x, rho, rho_mean, nfr = density_profile(frames, field, kind=kind,
                                            bins=bins, weights=weights)
    vext_eV = vext_of_x(x, field)
    kT = KB_EV_PER_K * field["T_K"]
    vext = vext_eV / kT            # PRL/PNAS convention: Vext in units of kT
    mask = rho > 1e-8
    mu = (float(np.mean(np.log(rho[mask]) + vext[mask]))
          if mu_override is None else float(mu_override))
    c1 = np.full(bins, np.nan)
    c1[mask] = np.log(rho[mask]) + vext[mask] - mu
    write_prl_f1(outdir / "f1", sim_id, rho, vext, mu)
    np.save(outdir / "f1" / f"c1_{sim_id}.npy", c1.astype(np.float32))
    np.save(outdir / "f1" / f"x_{sim_id}.npy", x.astype(np.float32))
    with open(outdir / "f1" / f"c1_{sim_id}.csv", "w", newline="") as fh:
        import csv as _csv
        wr = _csv.writer(fh)
        wr.writerow(["x_A", "rho_A3", "Vext_kT", "c1"])
        for xi, ri, vi, ci in zip(x, rho, vext, c1):
            wr.writerow([f"{xi:.6f}", f"{ri:.8e}", f"{vi:.8e}", f"{ci:.8e}"])

    if pnas_dict is not None:
        pnas_dict[sim_id] = {"rho": rho.astype(np.float32),
                             "Vext": vext.astype(np.float32)}

    rec = {
        "sim_id": sim_id, "species": species, "T_K": field["T_K"],
        "rho_mean_A3": rho_mean, "n_frames": nfr, "bins": bins,
        "density_kind": kind, "mu": mu,
        "mu_convention": ("override" if mu_override is not None
                          else "profile_mean"),
        "weights_source": weights_source,
        "units": {"length": "A", "density": "A^-3", "Vext": "kT",
                  "mu": "kT", "c1": "dimensionless"},
        "field_max_abs_eV": float(np.max(np.abs(field["A_eV"]))),
        "source": str(traj), "source_sha256": sha256(traj),
        "field_sha256": sha256(work / "field.json"),
    }

    if c2_on_field:
        if pair is None:
            pair = (elements[0], elements[0])
        if rmax is None:
            rmax = min(8.0, 0.5 * min(frames[0].cell.cellpar()[:3]))
        first = pair[0]
        syms = np.asarray(frames[0].get_chemical_symbols())
        rho_pair = int((syms == first).sum()) / frames[0].get_volume()
        r, g = rdf(frames, pair=pair, rmax=rmax, bins=c2_bins,
                   weights=weights)
        r, c2r, k, c2k = c2_from_g(r, g, rho_pair)
        write_prl_f2(outdir / "f2", sim_id, rho_pair, r, c2r, mu)
        np.savez(outdir / "f2" / f"c2k_{sim_id}.npz", r=r, g=g, k=k, c2k=c2k,
                 rho=rho_pair)
        rec.update({"c2_pair": list(pair), "c2_rmax_A": rmax,
                    "c2_bins": c2_bins, "rho_pair_A3": rho_pair,
                    "c2_short_trajectory": nfr < 200})
    return rec


def process_bulk(traj, outdir: Path, elements, species, T, density,
                 fmt="lammps-dump-text", stride=1, pair=None, rmax=None,
                 c2_bins=80, bins=2000):
    """Bulk trajectory -> flat F1 entry + F2 c2 entry."""
    traj = Path(traj)
    outdir = Path(outdir)
    frames = read_traj(traj, elements, stride=stride, fmt=fmt)
    if not frames:
        return None
    sim_id = f"bulk-{species}-r{density:g}-{T}K"
    vol = frames[0].get_volume()
    rho_flat = np.full(bins, float(density), dtype=np.float32)
    L = float(np.cbrt(vol))
    x = np.linspace(0.0, L, bins)
    write_prl_f1(outdir / "f1", sim_id, rho_flat, np.zeros(bins), 0.0)
    if pair is None:
        pair = (elements[0], elements[0])
    if rmax is None:
        rmax = min(8.0, 0.5 * min(frames[0].cell.cellpar()[:3]))
    syms = np.asarray(frames[0].get_chemical_symbols())
    rho_pair = int((syms == pair[0]).sum()) / vol
    r, g = rdf(frames, pair=pair, rmax=rmax, bins=c2_bins)
    r, c2r, k, c2k = c2_from_g(r, g, rho_pair)
    write_prl_f2(outdir / "f2", sim_id, rho_pair, r, c2r, 0.0)
    np.savez(outdir / "f2" / f"c2k_{sim_id}.npz", r=r, g=g, k=k, c2k=c2k,
             rho=rho_pair)
    return {
        "sim_id": sim_id, "species": species, "T_K": T,
        "density_mol_A3": density, "n_frames": len(frames), "bins": bins,
        "density_kind": "bulk", "c2_pair": list(pair),
        "c2_rmax_A": rmax, "c2_bins": c2_bins, "rho_pair_A3": rho_pair,
        "c2_short_trajectory": len(frames) < 200,
        "source": str(traj), "source_sha256": sha256(traj),
    }


def rdf(frames, pair=("O", "O"), rmax=8.0, bins=80, weights=None):
    edges = np.linspace(0.0, rmax, bins + 1)
    hist = np.zeros(bins)
    norm = 0.0
    for i, atoms in enumerate(frames):
        w = 1.0 if weights is None else float(weights[i])
        if w == 0.0:
            continue
        syms = np.asarray(atoms.get_chemical_symbols())
        ia = np.where(syms == pair[0])[0]
        ib = np.where(syms == pair[1])[0]
        dist = atoms.get_all_distances(mic=True)
        block = dist[np.ix_(ia, ib)]
        if pair[0] == pair[1]:
            iu = np.triu_indices(len(ia), k=1)
            rr = block[iu]
            expected = 0.5 * len(ia) * (len(ia) - 1)
        else:
            rr = block.reshape(-1)
            expected = len(ia) * len(ib)
        h, _ = np.histogram(rr, bins=edges)
        hist += w * h
        shell = 4.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
        norm += w * expected * shell / atoms.get_volume()
    g = hist / np.maximum(norm, 1e-30)
    return 0.5 * (edges[1:] + edges[:-1]), g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[],
                    help="site-field run directories (each with field.json "
                         "and traj.lammpstrj), or globs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--density", choices=["molecular", "atomic"],
                    default="molecular")
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--rmax", type=float, default=None)
    ap.add_argument("--c2-bins", type=int, default=80)
    ap.add_argument("--c2-pair", nargs=2, default=None)
    ap.add_argument("--c2-on-field", action="store_true",
                    help="also write F2 c2 entries for field-modulated runs")
    ap.add_argument("--weights-csv", default=None,
                    help="single reweight_frames.csv to apply to all runs")
    ap.add_argument("--weights-in-run", action="store_true",
                    help="look for <run>/reweight/reweight_frames.csv per run")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-pnas", action="store_true")
    ap.add_argument("--bulk-traj", default=None,
                    help="optional bulk trajectory (F2 c2 entry)")
    ap.add_argument("--bulk-elements", nargs="+", default=["O", "H"])
    ap.add_argument("--bulk-species", default="H2O")
    ap.add_argument("--bulk-T", type=float, default=300.0)
    ap.add_argument("--bulk-density", type=float, required=False,
                    help="molecules/A^3")
    ap.add_argument("--bulk-format", default="lammps-dump-text",
                    choices=["lammps-dump-text", "extxyz"])
    ap.add_argument("--bulk-stride", type=int, default=0,
                    help="defaults to --stride")
    args = ap.parse_args(argv)

    import glob as globmod
    works = []
    for pat in args.runs:
        hits = sorted(globmod.glob(pat))
        works.extend(Path(h) for h in (hits or [pat]))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    def load_weights(work: Path):
        import csv as _csv
        p = Path(args.weights_csv) if args.weights_csv else (
            work / "reweight" / "reweight_frames.csv"
            if args.weights_in_run else None)
        if p is None or not p.exists():
            return None, "none"
        with open(p, newline="") as fh:
            w = np.array([float(r["weight"]) for r in _csv.DictReader(fh)])
        return w, str(p)

    pnas = {}
    recs = []
    for work in works:
        if not (work / "field.json").exists():
            print(f"[skip] {work}: no field.json")
            continue
        weights, wsrc = load_weights(work)
        rec = process_run(work, outdir, bins=args.bins,
                          kind=args.density, mu_override=args.mu,
                          rmax=args.rmax, c2_bins=args.c2_bins,
                          pair=tuple(args.c2_pair) if args.c2_pair else None,
                          pnas_dict=None if args.no_pnas else pnas,
                          stride=args.stride, weights=weights,
                          c2_on_field=args.c2_on_field,
                          weights_source=wsrc)
        if rec is None:
            print(f"[skip] {work}: no trajectory")
            continue
        print(json.dumps(rec))
        recs.append(rec)
    if args.bulk_traj:
        rec = process_bulk(args.bulk_traj, outdir, args.bulk_elements,
                           args.bulk_species, args.bulk_T,
                           args.bulk_density, fmt=args.bulk_format,
                           stride=args.bulk_stride or args.stride,
                           rmax=args.rmax,
                           c2_bins=args.c2_bins,
                           pair=(tuple(args.c2_pair)
                                 if args.c2_pair else None))
        if rec:
            print(json.dumps(rec))
            recs.append(rec)
    if pnas:
        np.save(outdir / "pnas_sim_dict.npy", pnas)
        print(f"[pnas] wrote {len(pnas)} simulations to "
              f"{outdir / 'pnas_sim_dict.npy'}")

    # train/val split by simulation (whole trajectories, no frame leakage)
    import csv as _csv
    rng = np.random.default_rng(20261029)
    ids = [r["sim_id"] for r in recs]
    order = rng.permutation(len(ids))
    n_val = max(1, int(round(args.val_frac * len(ids)))) if len(ids) > 1 else 0
    val = {ids[i] for i in order[:n_val]}
    (outdir / "train_files.txt").write_text(
        "\n".join(i for i in ids if i not in val) + "\n")
    (outdir / "val_files.txt").write_text(
        "\n".join(i for i in ids if i in val) + "\n")
    with open(outdir / "index.csv", "w", newline="") as fh:
        wr = _csv.writer(fh)
        cols = ["sim_id", "species", "T_K", "rho_mean_A3", "n_frames",
                "mu", "mu_convention", "weights_source", "field_max_abs_eV",
                "c2_pair", "rho_pair_A3", "c2_short_trajectory", "split"]
        wr.writerow(cols)
        for r in recs:
            wr.writerow([r.get(c, "") if not isinstance(r.get(c), list)
                         else " ".join(map(str, r[c])) for c in cols[:-1]]
                        + ["val" if r["sim_id"] in val else "train"])
    (outdir / "manifest.json").write_text(json.dumps({
        "schema": "prl-f1/f2 + pnas sim_dict",
        "generated_by": "sampling.cdft_dataset",
        "bins": args.bins, "density": args.density,
        "val_frac": args.val_frac,
        "n_runs": len(recs),
        "runs": recs,
    }, indent=1))
    print(f"[done] {len(recs)} runs -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
