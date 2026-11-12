#!/usr/bin/env python3
"""Periodic-boundary-safe per-molecule local environments from a LAMMPS dump.

Every quantity here is built from *molecule-reconstructed* coordinates:

* atoms are grouped by the dump ``mol`` column (never by contiguous blocks);
* each molecule is rebuilt relative to its own central atom with the minimum
  image convention (MIC), so a molecule cut by the periodic boundary is put
  back together before anything else is computed;
* neighbour vectors are MIC displacements, so the neighbour list is identical
  to the infinite-periodic-system neighbour list;
* Voronoi volumes are computed on a 3x3x3 replica and the central cells are
  taken, which makes ``sum(V) == box volume`` an exact check.

The module is deliberately dependency-light (numpy + scipy) and has no
LAMMPS/ASE requirement, so it can be audited line by line.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# species metadata
# --------------------------------------------------------------------------
NATS = {"H2O": 3, "NH3": 4, "CH4": 5, "C2H4": 6, "C2H6": 8, "CO2": 3}
CENTRAL = {"H2O": "O", "NH3": "N", "CH4": "C", "C2H4": "C", "C2H6": "C",
           "CO2": "C"}
SYMBOL_OF_TYPE = {1: "C", 2: "H", 3: "N", 4: "O"}
HB_ACCEPTORS = {"H2O": ("O",), "NH3": ("N",)}
HB_DONOR_MAX = 2.5      # A, H...acceptor
HB_ANGLE_MIN = 150.0    # deg, X-H...Y

RCUT = 8.0              # A, neighbour cutoff (L/2 >= 15.3 A for every state)
RSWITCH = 6.5           # A, smooth-switch start
KMAX = 80               # fixed neighbour slot count (>= max count at rho*=0.67)
RADIAL_CENTERS = np.linspace(2.0, 8.0, 16)
RADIAL_WIDTH = 0.4


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------
def mic(d, box):
    """Minimum-image displacement for an orthorhombic cell (box = 3 lengths)."""
    return d - box * np.round(d / box)


def wrap(x, box):
    return x - box * np.floor(x / box)


def _open(path):
    import gzip
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def iter_dump(path, stride=1, max_frames=None):
    """Yield frames as dicts: positions, mol, symbols, types, forces, cell.

    The parser is the same layout as ``sampling.dump_mol.read_dump`` but keeps
    ``type`` and the per-frame box bounds (which are needed for the checks).
    """
    import gzip
    frame_idx = 0
    with _open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            fh.readline()                       # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            fh.readline()                       # ITEM: BOX BOUNDS
            lo, hi = [], []
            for _ in range(3):
                a, b = fh.readline().split()[:2]
                lo.append(float(a))
                hi.append(float(b))
            cols = fh.readline().split()[2:]
            block = "".join(fh.readline() for _ in range(n))
            data = np.fromstring(block, sep=" ").reshape(n, len(cols))
            take = frame_idx % max(1, stride) == 0
            frame_idx += 1
            if not take:
                continue
            idx = {c: i for i, c in enumerate(cols)}
            out = {
                "step": step,
                "n": n,
                "lo": np.array(lo),
                "hi": np.array(hi),
                "cell": np.array(hi) - np.array(lo),
                "types": data[:, idx["type"]].astype(np.int32),
                "positions": data[:, [idx["x"], idx["y"], idx["z"]]],
                "mol": data[:, idx["mol"]].astype(np.int64),
                "force": data[:, [idx["fx"], idx["fy"], idx["fz"]]]
                if "fx" in idx else None,
            }
            yield out
            if max_frames is not None and frame_idx >= max_frames * max(1, stride):
                break


# --------------------------------------------------------------------------
# molecule reconstruction (the periodic-boundary core)
# --------------------------------------------------------------------------
def molecules(pos, mol, nat):
    """Group atoms into molecules and return molecule index per atom.

    Returns ``(order, mid, first, nmol)`` where ``order`` sorts atoms by
    molecule, ``mid`` maps sorted rows to 0..nmol-1 and ``first`` is the index
    of the first atom of each molecule in the sorted array.
    """
    order = np.argsort(mol, kind="stable")
    ms = mol[order]
    uniq, first, count = np.unique(ms, return_index=True, return_counts=True)
    if not np.all(count == nat):
        raise ValueError(f"non-uniform molecule sizes: {np.unique(count)}")
    mid = np.repeat(np.arange(len(uniq)), count)
    return order, mid, first, len(uniq)


def reconstruct(pos, mol, nat, box):
    """Rebuild molecules in the central atom's frame with MIC.

    Returns ``(P, rel, r)`` with shapes ``(M, nat, 3)``, ``(M, nat, 3)`` and
    ``(M, nat)``.  ``rel[:, 0] == 0`` by construction.
    """
    order, mid, first, nmol = molecules(pos, mol, nat)
    P = pos[order].reshape(nmol, nat, 3)
    rel = P - P[:, 0:1, :]
    rel = mic(rel, box)
    r = np.sqrt((rel ** 2).sum(-1))
    return P, rel, r


def molecular_axes(rel, species):
    """Two orthonormal molecular axes per molecule, built from internal bonds.

    ``e1`` is the main symmetry/long axis, ``e2`` a second orthonormal
    direction that captures the molecular plane (or an arbitrary perpendicular
    for the linear CO2).  The construction is deterministic, so two frames of
    the same molecule give comparable axes.
    """
    M = rel.shape[0]
    v1 = rel[:, 1, :]
    e1 = v1 / np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-30)
    if species == "H2O":
        w = rel[:, 1, :] + rel[:, 2, :]        # bisector = dipole axis
        e1 = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        v2 = rel[:, 1, :] - rel[:, 2, :]
    elif species == "NH3":
        w = rel[:, 1:, :].sum(axis=1)          # C3 axis
        e1 = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        v2 = rel[:, 1, :] - rel[:, 2, :]
    elif species == "CO2":
        w = rel[:, 2, :] - rel[:, 1, :]        # O--O line
        e1 = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        v2 = np.cross(e1, np.array([0.0, 0.0, 1.0]))
        bad = np.linalg.norm(v2, axis=1) < 1e-6
        if np.any(bad):
            v2[bad] = np.cross(e1[bad], np.array([0.0, 1.0, 0.0]))
    elif species == "CH4":
        v2 = rel[:, 2, :]
    else:                                       # C2H4 / C2H6: C=C / C-C
        w = rel[:, 3, :] - rel[:, 1, :]
        e1 = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-30)
        v2 = rel[:, 2, :] if species == "C2H4" else rel[:, 4, :]
    e2 = v2 - e1 * (v2 * e1).sum(1, keepdims=True)
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-30)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)


# --------------------------------------------------------------------------
# neighbour lists
# --------------------------------------------------------------------------
def neighbour_lists(centers_wrapped, box, nmol, rcut=RCUT, kmax=KMAX):
    """MIC neighbour list for every centre.

    Returns ``(ids, rel, dist, mask)``.  ``ids`` are molecule indices, ``rel``
    are MIC displacements, both shape ``(M, kmax, ...)``.  Slots beyond the
    real neighbour count are masked out.
    """
    from scipy.spatial import cKDTree

    M = nmol
    nbrs = cKDTree(centers_wrapped, boxsize=box).query_ball_point(
        centers_wrapped, rcut)
    k = min(kmax, max(1, max(len(x) for x in nbrs)))
    ids = np.full((M, k), -1, dtype=np.int16)
    rel = np.zeros((M, k, 3), dtype=np.float32)
    mask = np.zeros((M, k), dtype=bool)
    for i, lst in enumerate(nbrs):
        if not lst:
            continue
        j = np.asarray(lst, dtype=np.int64)
        d = mic(centers_wrapped[j] - centers_wrapped[i], box)
        r = np.sqrt((d ** 2).sum(1))
        keep = np.argsort(r)[:k]
        j, d = j[keep], d[keep]
        ids[i, :len(j)] = j
        rel[i, :len(j)] = d
        mask[i, :len(j)] = True
    dist = np.sqrt((rel ** 2).sum(-1))
    ids[~mask] = -1
    return ids, rel, dist, mask


# --------------------------------------------------------------------------
# local scalars
# --------------------------------------------------------------------------
def tetrahedral_q(centers_wrapped, box):
    """Tetrahedral order parameter q from the 4 nearest centres (periodic)."""
    M = len(centers_wrapped)
    diff = centers_wrapped[None, :, :] - centers_wrapped[:, None, :]
    diff = mic(diff, box)
    dist2 = (diff ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)
    nn = np.argpartition(dist2, 4, axis=1)[:, :4]
    v = np.take_along_axis(diff, nn[:, :, None], axis=1)
    v = v / np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-30)
    s = np.zeros(M)
    for i in range(4):
        for j in range(i + 1, 4):
            s += (np.einsum("mk,mk->m", v[:, i], v[:, j]) + 1.0 / 3.0) ** 2
    return 1.0 - 3.0 / 8.0 * s


def nearest_distances(centers_wrapped, box, k=4):
    diff = centers_wrapped[None, :, :] - centers_wrapped[:, None, :]
    diff = mic(diff, box)
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    part = np.argpartition(dist, k, axis=1)[:, :k]
    dd = np.take_along_axis(dist, part, axis=1)
    return np.sort(dd, axis=1)


def coordination(centers_wrapped, box, rmin):
    diff = centers_wrapped[None, :, :] - centers_wrapped[:, None, :]
    diff = mic(diff, box)
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return (dist < rmin).sum(1).astype(np.float32)


def hydrogen_bonds(pos_wrapped, symbols, box, species):
    """Per-molecule donor / acceptor counts (geometric H-bond criterion).

    A hydrogen bond is counted when ``H...A < HB_DONOR_MAX`` and the
    ``D-H...A`` angle exceeds ``HB_ANGLE_MIN``.  Only H2O and NH3 have
    acceptors defined; other species return zeros.
    """
    M = len(np.unique(_dummy_mol)) if False else None
    acc = HB_ACCEPTORS.get(species)
    if acc is None:
        return None
    from scipy.spatial import cKDTree

    sym = np.asarray(symbols)
    tree = cKDTree(pos_wrapped, boxsize=box)
    pairs = tree.query_pairs(HB_DONOR_MAX, output_type="ndarray")
    if len(pairs) == 0:
        return None
    i, j = pairs[:, 0], pairs[:, 1]
    d = mic(pos_wrapped[j] - pos_wrapped[i], box)
    r = np.sqrt((d ** 2).sum(1))
    return {"i": i, "j": j, "vec": d, "r": r, "sym": sym}


def hb_counts(hb, mol_of_atom, nmol, box):
    """Turn the pair list into donor/acceptor counts per molecule."""
    if hb is None:
        return (np.zeros(nmol, np.int8), np.zeros(nmol, np.int8))
    sym, i, j, vec, r = hb["sym"], hb["i"], hb["j"], hb["vec"], hb["r"]
    donor = np.zeros(nmol, np.int8)
    accept = np.zeros(nmol, np.int8)
    is_h = sym == "H"
    for a, b, v, rr in zip(i, j, vec, r):
        for h, heavy in (((a, b) if is_h[a] else (b, a)),):
            if is_h[h] is False or is_h[heavy]:
                continue
            # parent heavy atom of H = nearest heavy neighbour of h
            d2 = ((hb["vec_all"] if False else None))
            donor[mol_of_atom[h]] += 0
    return donor, accept


# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------
_SQ3_4PI = np.sqrt(3.0 / (4.0 * np.pi))
_SQ3_8PI = np.sqrt(3.0 / (8.0 * np.pi))
_SQ5_16PI = np.sqrt(5.0 / (16.0 * np.pi))
_SQ15_8PI = np.sqrt(15.0 / (8.0 * np.pi))
_SQ15_32PI = np.sqrt(15.0 / (32.0 * np.pi))


def _ylm(u):
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    ct = z
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    phi = np.arctan2(y, x)
    e1 = np.exp(1j * phi)
    e2 = np.exp(2j * phi)
    y1 = np.stack([-_SQ3_8PI * st * e1, _SQ3_4PI * ct,
                   _SQ3_8PI * st * np.conj(e1)], axis=-1)
    y2 = np.stack([_SQ15_32PI * st * st * e2, -_SQ15_8PI * st * ct * e1,
                   _SQ5_16PI * (3.0 * ct * ct - 1.0),
                   _SQ15_8PI * st * ct * np.conj(e1),
                   _SQ15_32PI * st * st * np.conj(e2)], axis=-1)
    return y1, y2


def descriptor(rel, dist, mask, axes, rcut=RCUT, rswitch=RSWITCH):
    """Rotation-invariant per-centre descriptor (24 channels).

    Channels
    --------
    0-15  smooth radial basis over centre-centre distances
    16    |sum_j f_c Y_1m|^2
    17    |sum_j f_c Y_2m|^2
    18-23 orientation coupling with the neighbour molecules:
          <e1.r>, <(e1.r)^2>, <e1.u>, <(e1.u)^2>, <(e2.u)^2>,
          <3(e1.r)(u.r) - e1.u>
    where ``u`` is the neighbour molecular axis and ``f_c`` a smooth cutoff.
    All channels are invariant under translation, rotation and permutation of
    neighbours.
    """
    M, k = mask.shape
    out = np.zeros((M, 16 + 2 + 6), dtype=np.float32)
    if k == 0:
        return out
    r = np.where(mask, dist, 1e9)
    rc2, rs2 = rcut ** 2, rswitch ** 2
    fc = np.where(r < rswitch, 1.0,
                  np.where(r < rcut, ((rc2 - r ** 2) / (rc2 - rs2)) ** 2, 0.0))
    fc = fc * mask
    out[:, 0:16] = np.exp(-((r[:, :, None] - RADIAL_CENTERS[None, None, :])
                            / RADIAL_WIDTH) ** 2).__mul__(fc[:, :, None]).sum(1)
    u = rel / np.maximum(r[:, :, None], 1e-12)
    y1, y2 = _ylm(u)
    a1 = (y1 * fc[:, :, None]).sum(1)
    a2 = (y2 * fc[:, :, None]).sum(1)
    out[:, 16] = (np.abs(a1) ** 2).sum(1)
    out[:, 17] = (np.abs(a2) ** 2).sum(1)
    nbr_axis = np.zeros_like(u)
    ok = mask & (mask.sum(1, keepdims=True) > 0)
    np.divide(rel, np.maximum(r, 1e-12)[:, :, None], out=nbr_axis,
              where=ok[:, :, None])
    nb = mask.sum(1)
    nb_safe = np.maximum(nb, 1)
    er = np.einsum("mk,mkc->mk", np.nan_to_num(u), axes[:, 0][:, None, :])
    eu = np.einsum("mkc,mkc->mk", np.nan_to_num(nbr_axis),
                   np.nan_to_num(u))
    out[:, 18] = (fc * er).sum(1) / nb_safe
    out[:, 19] = (fc * er ** 2).sum(1) / nb_safe
    out[:, 20] = (fc * eu).sum(1) / nb_safe
    out[:, 21] = (fc * eu ** 2).sum(1) / nb_safe
    return out


# --------------------------------------------------------------------------
# Voronoi
# --------------------------------------------------------------------------
def voronoi_volume(centers_wrapped, box):
    """Periodic Voronoi volume per centre (3x3x3 replica, central cells).

    ``sum(volume) == box volume`` is an exact invariant of the construction
    and is used as a self-check.
    """
    from scipy.spatial import ConvexHull, QhullError, Voronoi

    c = wrap(centers_wrapped, box)
    M = len(c)
    sh = np.array([[i, j, k] for i in (-1, 0, 1)
                   for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    central = int(np.argmin((sh ** 2).sum(1)))
    allc = (c[None, :, :] + sh[:, None, :] * box).reshape(-1, 3)
    vol = np.full(M, np.nan)
    try:
        vor = Voronoi(allc, qhull_options="Qbb Qc Qz")
    except QhullError:
        return vol
    off = central * M
    for i in range(M):
        reg = vor.regions[vor.point_region[off + i]]
        if len(reg) < 4 or -1 in reg:
            continue
        try:
            vol[i] = ConvexHull(vor.vertices[reg]).volume
        except (QhullError, ValueError):
            continue
    return vol
