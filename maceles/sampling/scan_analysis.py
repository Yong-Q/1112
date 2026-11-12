#!/usr/bin/env python3
"""Post-processing for a MACELES bulk density scan.

Two stages, both resumable:

1. ``--stage state`` (default when ``--root`` given): for every completed
   state under ``<root>/**/samples/<sim_id>/`` extract
       - c1/c2 via ``sampling.les_c1c2.analyze_trajectory``
       - structural/orientational observables (this module)
   into ``<state>/c1c2`` and ``<state>/analysis``.

2. ``--stage aggregate``: collect every ``analysis/state.json`` and
   ``c1c2/f2`` payload into
       ``<out>/scan_summary.csv``          one row per state
       ``<out>/c2_<species>.npz``          r grid + c2 matrix + rho/T
       ``<out>/figures/*.png``             c2(r), g(r), trends

Usage
-----
    python -m sampling.scan_analysis --root bench/maceles_density_scan \
        --out bench/maceles_density_scan/analysis_scan
    python -m sampling.scan_analysis --stage aggregate --root ... --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ELEMENTS = ["C", "H", "N", "O"]
NATS = {"H2O": 3, "NH3": 4, "CH4": 5, "C2H4": 6, "C2H6": 8, "CO2": 3}
CENTRAL = {"H2O": "O", "NH3": "N", "CH4": "C", "C2H4": "C", "C2H6": "C",
           "CO2": "C"}
HB_ACCEPTORS = {"H2O": ("O",), "NH3": ("N",)}
HB_DONOR_MAX = 2.5          # A, H...acceptor
HB_ANGLE_MIN = 150.0        # deg, X-H...Y
RDF_RMAX = 8.0
RDF_BINS = 100


# ----------------------------------------------------------------------------
# frame access
# ----------------------------------------------------------------------------
def iter_frames(path, stride=1):
    # numpy reader: ASE re-scans the gzip stream per frame and is unusably
    # slow on the multi-hundred-MB production dumps.
    from ase import Atoms
    from sampling.dump_mol import read_dump
    for f in read_dump(path, elements=ELEMENTS, stride=max(1, stride)):
        yield Atoms(symbols=f["symbols"], positions=f["positions"],
                    cell=f["cell"], pbc=True)


def n_frames_of(state: Path, traj: Path) -> int:
    meta = state / "meta.json"
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            steps = int(d.get("steps") or 0)
            every = int(d.get("dump_every") or 0)
            if steps and every:
                return max(1, steps // every)
        except Exception:
            pass
    if traj.suffix == ".gz":
        import gzip
        with gzip.open(traj, "rt", errors="ignore") as fh:
            return sum(1 for ln in fh if ln.startswith("ITEM: TIMESTEP"))
    with open(traj, "rt", errors="ignore") as fh:
        return sum(1 for ln in fh if ln.startswith("ITEM: TIMESTEP"))


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def _mic(vec, cell):
    return vec - cell * np.round(vec / cell)


def _orientation_vec(rel, species):
    """Unit molecular orientation vector (per species), built from atom
    positions relative to the central (heavy) atom, already min-image'd.

    - H2O/NH3 : dipole / symmetry axis = sum of X-H bond vectors
                (C2/C3 axis pointing from the central atom toward the H's).
    - CO2     : molecular axis = O--O line.
    - CH4/C2H4/C2H6 : first bond (C-H / C=C / C-C) as the molecular axis.

    Raw bond vectors keep a fixed direction, so there is no sign ambiguity
    from the eigendecomposition.
    """
    if species in ("H2O", "NH3"):
        v = rel[:, 1:, :].sum(axis=1)
    elif species == "CO2":
        v = rel[:, 1, :] - rel[:, 2, :]
    else:
        v = rel[:, 1, :]
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-30)


def _tetrahedral_q(centrals, cell):
    """q for the 4 nearest central atoms (periodic)."""
    diff = centrals[None, :, :] - centrals[:, None, :]
    diff = _mic(diff, cell)
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, 1e9)
    nn = np.argsort(dist, axis=1)[:, :4]
    out = np.empty(len(centrals))
    for a in range(len(centrals)):
        v = diff[a, nn[a]]
        v /= np.linalg.norm(v, axis=1)[:, None]
        s = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                s += (np.dot(v[i], v[j]) + 1.0 / 3.0) ** 2
        out[a] = 1.0 - 3.0 / 8.0 * s
    return out


def _rdf(dist, mask_a, mask_b, volume, rmax=RDF_RMAX, bins=RDF_BINS):
    ia, ib = np.where(mask_a)[0], np.where(mask_b)[0]
    block = dist[np.ix_(ia, ib)]
    if mask_a is mask_b:
        iu = np.triu_indices(len(ia), k=1)
        rr = block[iu]
        expected = 0.5 * len(ia) * (len(ia) - 1)
    else:
        rr = block.reshape(-1)
        expected = len(ia) * len(ib)
    edges = np.linspace(0, rmax, bins + 1)
    h, _ = np.histogram(rr, bins=edges)
    shell = 4 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3
    g = h / np.maximum(expected * shell / volume, 1e-30)
    return 0.5 * (edges[1:] + edges[:-1]), g


def _hbonds(atoms, sym, acceptors):
    """Number of X-H...Y hydrogen bonds in the frame (O(N * neighbours))."""
    from ase.neighborlist import neighbor_list
    ii, jj, dd, dvec = neighbor_list("ijdD", atoms, HB_DONOR_MAX)
    z = atoms.numbers
    nb = {}
    for a, b, d, v in zip(ii, jj, dd, dvec):
        nb.setdefault(int(a), []).append((int(b), float(d), v))
    acc = set(acceptors)
    hb = 0
    for h in np.where(sym == "H")[0]:
        nbrs = nb.get(int(h), ())
        heavy = [(b, d, v) for b, d, v in nbrs if z[b] != 1]
        if not heavy:
            continue
        parent, _, vh = min(heavy, key=lambda t: t[1])
        nvh = np.linalg.norm(vh)
        for b, _, v in nbrs:
            if b == parent or sym[b] not in acc:
                continue
            cosv = np.dot(vh, v) / (nvh * np.linalg.norm(v))
            if np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))) > HB_ANGLE_MIN:
                hb += 1
    return hb



MASS = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}


def com_timeseries(traj, n_mol, natoms_per_mol):
    """Per-frame mass-weighted centre of mass plus the COM mean-square displacement.

    Uses the ``mol`` column when the dump has one (so repeated elements such as
    the two H of water or the six H of ethane are grouped correctly); otherwise
    falls back to contiguous atom blocks.
    """
    from sampling.dump_mol import read_dump
    com, unwrapped, box = [], [], None
    for fr in read_dump(traj, elements=ELEMENTS):
        sym = np.array([str(x) for x in fr["symbols"]])
        pos = np.asarray(fr["positions"])
        mol = fr.get("mol")
        if mol is None:
            mol = np.repeat(np.arange(n_mol), natoms_per_mol)
        umol, inv = np.unique(mol, return_inverse=True)
        m = np.array([MASS.get(x, 1.0) for x in sym])
        wsum = np.bincount(inv, weights=m)
        c = np.stack([np.bincount(inv, weights=m * pos[:, k]) for k in range(3)],
                     axis=1) / wsum[:, None]
        if box is None:
            box = np.asarray(fr["cell"], dtype=float)
        if unwrapped:
            d = c - com[-1]
            d -= box * np.round(d / box)
            c_un = unwrapped[-1] + d
        else:
            c_un = c.copy()
        com.append(c)
        unwrapped.append(c_un)
    com = np.asarray(com, dtype=np.float32)
    un = np.asarray(unwrapped)
    nf = len(un)
    lags = np.unique(np.linspace(0, nf - 1, min(nf, 40)).astype(int))
    msd = np.array([[lag, float(((un[lag:] - un[:nf - lag]) ** 2).sum(-1).mean())]
                    for lag in lags])
    return com, msd

# ----------------------------------------------------------------------------
# per-state analysis
# ----------------------------------------------------------------------------
def analyze_state(traj, species, outdir, n_frames=200, rdf_frames=20,
                  hb_frames=50, rng_seed=0, pairs=True, pair_frames=20,
                  com=True, timestep_ps=0.0005, dump_every=1000,
                  intra=True, intra_frames=200):
    """Return a dict of scalars; also writes ``analysis/profiles.npz``."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    nat = NATS[species]
    central_sym = CENTRAL[species]
    total = n_frames_of(Path(traj).parent.parent, Path(traj))
    stride = max(1, total // max(1, n_frames))
    rdf_stride = max(1, total // max(1, rdf_frames))
    hb_stride = max(1, total // max(1, hb_frames))

    bonds, angles, p2, qtet, hb_per_mol = [], [], [], [], []
    orient_axis, otheta, ophi = [], [], []
    r_pair = g_pair = r_xh = g_xh = None
    n_seen = 0
    n_rdf_seen = 0
    for idx, atoms in enumerate(iter_frames(traj, stride=1)):
        if idx % stride:
            continue
        sym = np.array(atoms.get_chemical_symbols())
        cell = np.array(atoms.cell.lengths())
        pos = atoms.get_positions().reshape(-1, nat, 3)
        if str(sym[0]) != central_sym:
            raise ValueError(f"{species}: first atom of a block is {sym[0]}, "
                             f"expected {central_sym}")
        rel = pos - pos[:, 0:1, :]
        rel = _mic(rel, cell)
        r = np.linalg.norm(rel, axis=2)
        bonds.append(r[:, 1:].ravel())
        # angles between the first two X-H (or X-Y) bonds
        if nat >= 3:
            v1, v2 = rel[:, 1, :], rel[:, 2, :]
            c = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) *
                                    np.linalg.norm(v2, axis=1))
            angles.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        ov = _orientation_vec(rel, species)
        p2.append(ov[:, 2] ** 2)
        orient_axis.append(ov)
        otheta.append(np.arccos(np.clip(ov[:, 2], -1.0, 1.0)))
        ophi.append(np.arctan2(ov[:, 1], ov[:, 0]))
        qtet.append(_tetrahedral_q(pos[:, 0, :], cell))
        if species in HB_ACCEPTORS and idx % hb_stride == 0:
            hb_per_mol.append(_hbonds(atoms, sym, HB_ACCEPTORS[species])
                              / len(pos))
        if idx % rdf_stride == 0:
            dist = atoms.get_all_distances(mic=True)
            m_central = sym == central_sym
            m_h = sym == "H"
            vol = atoms.get_volume()
            rp, gp = _rdf(dist, m_central, m_central, vol)
            rx, gx = _rdf(dist, m_central, m_h, vol)
            if r_pair is None:
                r_pair, g_pair = rp, gp.copy()
                r_xh, g_xh = rx, gx.copy()
            else:
                g_pair += gp
                g_xh += gx
            n_rdf_seen += 1
        n_seen += 1

    if n_seen == 0:
        raise RuntimeError(f"no frames read from {traj}")
    bonds = np.concatenate(bonds)
    p2 = np.concatenate(p2)
    orient_axis = np.concatenate(orient_axis)
    otheta = np.concatenate(otheta)
    ophi = np.concatenate(ophi)
    qtet = np.concatenate(qtet)
    g_pair = g_pair / max(1, n_rdf_seen)
    g_xh = g_xh / max(1, n_rdf_seen)

    i = int(np.argmax(g_pair))
    r_peak, g_peak = float(r_pair[i]), float(g_pair[i])
    # coordination to the first minimum after the peak
    tail = np.where(g_pair[i:] < 1.0)[0]
    r_cut = float(r_pair[i + tail[0]]) if len(tail) else float(r_pair[-1])
    shell = (r_pair > 0) & (r_pair < r_cut)
    dens = len(pos) / float(atoms.get_volume())
    coord = float(np.trapezoid(4 * np.pi * r_pair[shell] ** 2 * dens *
                               g_pair[shell], r_pair[shell])) if hasattr(
        np, "trapezoid") else float(np.trapz(
            4 * np.pi * r_pair[shell] ** 2 * dens * g_pair[shell],
            r_pair[shell]))

    rec = {
        "n_frames_used": int(n_seen),
        "stride": int(stride),
        "n_molecules": int(len(pos)),
        "bond_mean_A": float(bonds.mean()),
        "bond_std_A": float(bonds.std()),
        "angle_mean_deg": float(np.concatenate(angles).mean())
        if angles else None,
        "angle_std_deg": float(np.concatenate(angles).std())
        if angles else None,
        "p2_mean": float(p2.mean()),
        "q_tet_mean": float(qtet.mean()),
        "q_tet_std": float(qtet.std()),
        "hb_per_molecule": float(np.mean(hb_per_mol)) if hb_per_mol else None,
        "g_peak": g_peak,
        "r_peak_A": r_peak,
        "r_first_min_A": r_cut,
        "coordination": coord,
        "pair_label": f"{central_sym}-{central_sym}",
    }
    if pairs:
        from sampling.pair_correlations import pair_correlations
        pframes = []
        pstride = max(1, total // max(1, pair_frames))
        from sampling.dump_mol import read_dump
        try:
            for fr in read_dump(traj, elements=ELEMENTS, stride=pstride,
                                max_frames=pair_frames):
                pframes.append(fr)
        except Exception:
            pframes = []
            for idx, atoms in enumerate(iter_frames(traj, 1)):
                if idx % pstride == 0:
                    pframes.append(atoms)
                if len(pframes) >= pair_frames:
                    break
        pc = pair_correlations(pframes, rmax=RDF_RMAX, bins=RDF_BINS)
        np.savez(out / "pairs.npz", **{k: v for k, v in pc.items()})
        rec["pairs_file"] = "pairs.npz"
        rec["pairs"] = sorted(k[2:] for k in pc if k.startswith("g_"))
        rec["anisotropy_l1_max"] = float(max(
            [np.max(v) for k, v in pc.items() if k.startswith("a1_")] or [0.0]))
        rec["anisotropy_l2_max"] = float(max(
            [np.max(v) for k, v in pc.items() if k.startswith("a2_")] or [0.0]))

    if intra:
        try:
            from sampling.intramolecular import intramolecular_stats
            iframes = []
            istride = max(1, total // max(1, intra_frames))
            for idx, atoms in enumerate(iter_frames(traj, 1)):
                if idx % istride == 0:
                    iframes.append(atoms)
                if len(iframes) >= intra_frames:
                    break
            iv = intramolecular_stats(iframes, nat, max_frames=intra_frames)
            np.savez(out / "intra.npz", **iv)
            rec["intra_mean"] = {k: float(np.mean(v)) for k, v in iv.items()
                                 if k.startswith(("bond_", "angle_",
                                                  "dihedral_"))}
            rec["axes_p2_mean"] = float(iv.get("p2_mean", 0.0))
            rec["intra_frames"] = int(iv.get("nframes", 0))
        except Exception as exc:                     # noqa: BLE001
            rec["intra_error"] = f"{type(exc).__name__}: {exc}"

    if com:
        try:
            com_xyz, msd = com_timeseries(traj, int(len(pos)), nat)
            dt_ps = timestep_ps * dump_every
            np.savez(out / "com.npz", com=com_xyz,
                     lag_ps=msd[:, 0] * dt_ps, msd_A2=msd[:, 1])
            rec["com_frames"] = int(len(com_xyz))
            rec["com_nmol"] = int(com_xyz.shape[1])
            rec["msd_file"] = "com.npz"
        except Exception as exc:                     # noqa: BLE001
            rec["com_error"] = f"{type(exc).__name__}: {exc}"

    np.savez(out / "profiles.npz", r_pair=r_pair, g_pair=g_pair,
             r_xh=r_xh, g_xh=g_xh, bonds=bonds, p2=p2, q_tet=qtet,
             orient_axis=orient_axis, orient_theta=otheta, orient_phi=ophi)
    (out / "state.json").write_text(json.dumps(rec, indent=2))
    return rec


# ----------------------------------------------------------------------------
# discovery + aggregation
# ----------------------------------------------------------------------------
def _parse_sim_id(sid):
    """H2O-PRL-T1003-r07-0.012345 -> (species, T, rho)."""
    if "__" in sid:                      # partial_attempt-04/shard1__H2O-...
        sid = sid.split("__", 1)[1]
    sp = sid.split("-")[0]
    temp = rho = 0.0
    m = re.search(r"-T([0-9.]+)-", sid)
    if m:
        temp = float(m.group(1))
    m = re.search(r"-r\d+-([0-9.]+)$", sid)
    if m:
        rho = float(m.group(1))
    return sp, temp, rho


def discover_states(root: Path):
    """Yield (sim_id, species, temperature, density, state_dir, traj).

    A finished state keeps its dump in ``run/`` only until the archive loop
    moves it to ``traj_archive/<sim_id>.dump.gz`` and drops the worker copy;
    look in both places so analysis does not depend on that race.
    """
    root = Path(root)
    archive = root / "traj_archive"
    entries = []
    for traj in sorted(root.rglob("run/traj.dump*")):
        if "samples_old" in str(traj) or "_legacy" in str(traj):
            continue
        entries.append((traj.parent.parent, traj))
    if archive.is_dir():
        for f in sorted(archive.glob("*.dump.gz")):
            sid_arch = f.name[:-len(".dump.gz")]
            cands = [p for p in root.glob(f"shard_*/attempt-*/samples/{sid_arch}")
                     if p.is_dir() and "samples_old" not in str(p)]
            if not cands:
                cands = [p for p in root.glob(f"samples/{sid_arch}") if p.is_dir()]
            if cands:
                entries.append((sorted(cands)[-1], f))
    seen = set()
    for state, traj in entries:
        sid = state.name
        if sid in seen:
            continue
        seen.add(sid)
        sp, temp, rho = _parse_sim_id(sid)
        meta = state / "meta.json"
        if meta.exists():
            try:
                d = json.loads(meta.read_text())
                sp = d.get("species") or sp
                temp = float(d.get("temperature_K") or temp)
                rho = float(d.get("density_mol_A3") or rho)
            except Exception:
                pass
        if sp not in NATS:
            continue
        yield (sid, sp, temp, rho, state, traj)


def run_states(root: Path, outdir: Path, force=False, n_frames=200,
               c2_stride=10, rmax=None, c2_bins=80, verbose=True,
               pairs=True, pair_frames=20, prune_traj=False,
               require_meta=False):
    from sampling.les_c1c2 import analyze_trajectory
    rows = []
    for sid, species, temp, rho, state, traj in discover_states(root):
        ana = outdir / sid
        ana.mkdir(parents=True, exist_ok=True)
        rec_path = ana / "analysis.json"
        if require_meta and not (state / "meta.json").exists():
            continue
        if rec_path.exists() and not force:
            rows.append(json.loads(rec_path.read_text()))
            if verbose:
                print(f"[scan] skip {sid} (analysis exists)")
            continue
        if verbose:
            print(f"[scan] {sid}: {traj}")
        meta_p = state / "meta.json"
        ts, de = 0.0005, 1000
        if meta_p.exists():
            try:
                _m = json.loads(meta_p.read_text())
                ts = float(_m.get("timestep_ps") or ts)
                de = int(_m.get("dump_every") or de)
            except Exception:
                pass
        struct = analyze_state(traj, species, ana / "analysis",
                               n_frames=n_frames, pairs=pairs,
                               pair_frames=pair_frames, com=True,
                               timestep_ps=ts, dump_every=de)
        c1c2 = ana / "c1c2"
        try:
            c1, dens = analyze_trajectory(
                traj, c1c2, species, ELEMENTS, temp, density=rho or None,
                pair=tuple(species_pair(species)), rmax=rmax,
                c2_bins=c2_bins, stride=c2_stride)
        except Exception as exc:                    # keep going
            c1, dens = {"error": f"{type(exc).__name__}: {exc}"}, rho
        rec = {"sim_id": sid, "species": species, "temperature_K": temp,
               "density_mol_A3": dens, "traj": str(traj),
               "structure": struct, "c1c2": c1,
               "c1c2_dirs": {"f1": str(c1c2 / "f1"),
                             "f2": str(c1c2 / "f2")}}
        rec_path.write_text(json.dumps(rec, indent=2, default=str))
        if prune_traj:
            try:
                Path(traj).unlink()
                print(f"[scan] pruned {traj}")
            except OSError as exc:
                print(f"[scan] could not prune {traj}: {exc}")
        rows.append(rec)
    return rows


def species_pair(species):
    c = CENTRAL.get(species, "C")
    return (c, "O") if species == "CO2" else (c, c)


def aggregate(outdir: Path, rho_c=None):
    outdir = Path(outdir)
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for rec_path in sorted(outdir.rglob("analysis.json")):
        try:
            rec = json.loads(rec_path.read_text())
            rec["_dir"] = str(rec_path.parent)
            rows.append(rec)
        except Exception:
            continue
    if not rows:
        print("[scan] no analysis.json found")
        return []
    # --- summary table
    cols = ["sim_id", "species", "temperature_K", "density_mol_A3",
            "n_frames_used", "bond_mean_A", "angle_mean_deg", "q_tet_mean",
            "hb_per_molecule", "p2_mean", "r_peak_A", "g_peak",
            "r_first_min_A", "coordination"]
    with open(outdir / "scan_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            s = r.get("structure", {})
            w.writerow([r.get("sim_id"), r.get("species"),
                        r.get("temperature_K"), r.get("density_mol_A3")]
                       + [s.get(c) for c in cols[4:]])
    # --- c2 matrices per species
    series = {}
    for r in rows:
        sp = r.get("species")
        dirs = r.get("c1c2_dirs") or {}
        f2 = Path(dirs.get("f2") or
                  (Path(r.get("_dir", ".")) / "c1c2" / "f2"))
        if not f2.exists():
            continue
        r_file = next(iter(f2.glob("r_values_*.npy")), None)
        c_file = next(iter(f2.glob("c2_*.npy")), None)
        g_file = next(iter(f2.glob("c2k_*.npz")), None)
        if not (r_file and c_file):
            continue
        rgrid = np.load(r_file)
        c2 = np.load(c_file)
        g = np.load(g_file)["g"] if g_file else None
        series.setdefault(sp, []).append(
            (float(r.get("density_mol_A3") or 0.0),
             float(r.get("temperature_K") or 0.0), rgrid, c2, g,
             r.get("sim_id")))
    for sp, items in series.items():
        items.sort()
        r = items[0][2]
        c2 = np.vstack([it[3] for it in items])
        rho = np.array([it[0] for it in items])
        temp = np.array([it[1] for it in items])
        gg = np.vstack([it[4] for it in items]) if items[0][4] is not None \
            else None
        np.savez(outdir / f"c2_{sp}.npz", r=r, c2=c2, c2k_g=gg, rho=rho,
                 temperature=temp,
                 sim_id=np.array([it[5] for it in items], dtype=object))
    make_figures(outdir, rows, series, rho_c or {})
    print(f"[scan] {len(rows)} states -> {outdir}")
    return rows


def make_figures(outdir, rows, series, rho_c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) c2(r) per species, coloured by reduced density
    for sp, items in series.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap("viridis")
        rhos = [it[0] for it in items]
        lo, hi = min(rhos), max(rhos)
        for rho, temp, r, c2, g, sid in items:
            frac = 0.0 if hi == lo else (rho - lo) / (hi - lo)
            ax.plot(r, c2, color=cmap(frac), lw=1.4,
                    label=f"rho={rho:.5f}")
        ax.set_xlabel("r (A)")
        ax.set_ylabel("c2(r)")
        ax.set_title(f"{sp}: direct correlation vs density")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        if len(items) <= 10:
            ax.legend(fontsize=6, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(outdir / "figures" / f"c2_{sp}.png", dpi=150)
        plt.close(fig)

        if items[0][4] is not None:
            fig, ax = plt.subplots(figsize=(8, 5))
            for rho, temp, r, c2, g, sid in items:
                frac = 0.0 if hi == lo else (rho - lo) / (hi - lo)
                ax.plot(r, g, color=cmap(frac), lw=1.2)
            ax.set_xlabel("r (A)")
            ax.set_ylabel(f"g(r), {CENTRAL[sp]}-{CENTRAL[sp]}")
            ax.set_title(f"{sp}: pair distribution vs density")
            fig.tight_layout()
            fig.savefig(outdir / "figures" / f"gr_{sp}.png", dpi=150)
            plt.close(fig)

    # 2) trends vs reduced density
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [("g_peak", "first-peak height g(r)"),
               ("coordination", "coordination at first minimum"),
               ("q_tet_mean", "tetrahedral order q"),
               ("hb_per_molecule", "H-bonds per molecule")]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        for sp in sorted({r["species"] for r in rows}):
            pts = []
            for r in rows:
                if r["species"] != sp:
                    continue
                v = r.get("structure", {}).get(key)
                if v is None:
                    continue
                rr = r.get("density_mol_A3") or 0.0
                rc = rho_c.get(sp) or 1.0
                pts.append((rr / rc, v))
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3,
                    lw=1.2, label=sp)
        ax.set_xlabel("rho / rho_c")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "trends.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="scan root containing **/samples/<sim_id>/")
    ap.add_argument("--out", default=None,
                    help="analysis root (default <root>/analysis_scan)")
    ap.add_argument("--stage", choices=["all", "state", "aggregate"],
                    default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--c2-stride", type=int, default=10)
    ap.add_argument("--c2-bins", type=int, default=80)
    ap.add_argument("--rmax", type=float, default=None)
    ap.add_argument("--no-pairs", action="store_true",
                    help="skip the all-pair g(r)/anisotropy block")
    ap.add_argument("--pair-frames", type=int, default=20)
    ap.add_argument("--prune-traj", action="store_true",
                    help="delete each trajectory once its analysis is written")
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis_scan"
    rho_c = {}
    manifest = root / "density_scan.csv"
    if manifest.exists():
        import csv as _csv
        for row in _csv.DictReader(open(manifest)):
            rho_c.setdefault(row["species"], float(row["density"]) /
                             float(row.get("rho_c") or 1.0))
    if args.stage in ("all", "state"):
        run_states(root, out, force=args.force, n_frames=args.frames,
                   c2_stride=args.c2_stride, rmax=args.rmax,
                   c2_bins=args.c2_bins, pairs=not args.no_pairs,
                   pair_frames=args.pair_frames,
                   prune_traj=args.prune_traj)
    if args.stage in ("all", "aggregate"):
        aggregate(out, rho_c=rho_c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
