#!/usr/bin/env python3
"""Fast LAMMPS dump reader that keeps the ``mol`` column.

ASE's ``lammps-dump-text`` reader drops unknown columns, so molecule ids are
lost.  The MD dumps here are written as ``id mol type x y z fx fy fz``; this
reader returns exactly those fields so post-processing can group atoms by
molecule instead of assuming contiguous blocks.

    for frame in read_dump("traj.dump.gz", elements=["C","H","N","O"]):
        frame["symbols"], frame["positions"], frame["mol"], frame["forces"]
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

SYMBOL_OF_TYPE = {1: "C", 2: "H", 3: "N", 4: "O"}


def _open(path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def read_dump(path, elements=None, stride=1, max_frames=None):
    """Yield dicts with symbols, positions, mol ids, forces and the cell."""
    order = elements or ["C", "H", "N", "O"]
    frame_idx = 0
    with _open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            fh.readline()                       # timestep value
            fh.readline()                       # ITEM: NUMBER OF ATOMS
            n = int(fh.readline())
            fh.readline()                       # ITEM: BOX BOUNDS
            lows, highs = [], []
            for _ in range(3):
                lo, hi = fh.readline().split()[:2]
                lows.append(float(lo))
                highs.append(float(hi))
            cols = fh.readline().split()[2:]    # ITEM: ATOMS ...
            rows = [fh.readline() for _ in range(n)]
            data = np.array(" ".join(rows).split(), dtype=float)
            data = data.reshape(n, len(cols))
            take = frame_idx % max(1, stride) == 0
            frame_idx += 1
            if not take:
                continue
            idx = {c: i for i, c in enumerate(cols)}
            types = data[:, idx["type"]].astype(int)
            sym = [SYMBOL_OF_TYPE.get(t, order[min(t - 1, len(order) - 1)])
                   for t in types]
            out = {
                "symbols": np.array(sym),
                "positions": data[:, [idx["x"], idx["y"], idx["z"]]],
                "cell": np.array(highs) - np.array(lows),
            }
            if "mol" in idx:
                out["mol"] = data[:, idx["mol"]].astype(np.int64)
            if "fx" in idx:
                out["forces"] = data[:, [idx["fx"], idx["fy"], idx["fz"]]]
            yield out
            if max_frames is not None and frame_idx >= max_frames * max(1, stride):
                break
