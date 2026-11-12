#!/usr/bin/env python3
"""Post-process one shard's completed MD trajectories into scan/post.

For every manifest row whose MD finished (``scan/traj/<sim_id>/meta.json``
exists with a status of md_complete/complete) this extracts F1/F2 into
``<post-root>/<sim_id>/`` via ``sampling.les_c1c2.analyze_trajectory`` and
drops a ``les_c1c2.json`` marker so re-runs are no-ops.

Usage:
    python scripts/postprocess_shard.py \
        --manifest scan/shard_1/attempt-01/manifest.csv \
        --traj-root scan/traj --post-root scan/post
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path("/publicfs10/fs10-m9/home/m9s005258/qy/cdft/new")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sampling.les_c1c2 import (  # noqa: E402
    analyze_trajectory, parse_elements, parse_pair, _jsonable,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--traj-root", required=True)
    ap.add_argument("--post-root", required=True)
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--c2-bins", type=int, default=80)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args(argv)

    rows = list(csv.DictReader(open(args.manifest)))
    traj_root = Path(args.traj_root)
    post_root = Path(args.post_root)
    done = 0
    for row in rows:
        sim_id = row["sim_id"].strip()
        marker = post_root / sim_id / "les_c1c2.json"
        if marker.exists():
            done += 1
            continue
        meta_path = traj_root / sim_id / "meta.json"
        if not meta_path.exists():
            print(f"[post] skip {sim_id}: no meta.json (MD not done)")
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("status") not in ("md_complete", "complete"):
            print(f"[post] skip {sim_id}: status={meta.get('status')}")
            continue
        traj = meta.get("traj")
        if not traj or not Path(traj).exists():
            print(f"[post] ERROR {sim_id}: traj missing {traj}")
            continue
        species = row["species"].strip()
        elements = meta.get("elements") or parse_elements(
            row["elements"].split())
        pair = parse_pair(row["c2_pair"].split())
        outdir = post_root / sim_id
        rec, density = analyze_trajectory(
            traj, outdir, species, list(elements), float(meta["temperature_K"]),
            density=float(meta.get("density_mol_A3", 0.0) or 0.0),
            pair=pair, c2_bins=args.c2_bins, bins=args.bins, stride=args.stride)
        payload = {
            "schema": "les-c1c2-v1",
            "generated_by": "cdft.scripts.postprocess_shard",
            "mode": "postprocess",
            "sim_id": sim_id,
            "elements": list(elements),
            "c2_pair": list(pair),
            "traj": str(Path(traj).resolve()),
            "c1c2": rec,
            "outputs": {
                "f1_dir": str((outdir / "f1").resolve()),
                "f2_dir": str((outdir / "f2").resolve()),
            },
        }
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(_jsonable(payload), indent=1) + "\n")
        done += 1
        print(f"[post] {sim_id}: F1/F2 -> {outdir} "
              f"(frames={rec['n_frames']}, rmax={rec['c2_rmax_A']:.3g} A)")
    print(f"[post] {done}/{len(rows)} rows posted (or already done)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
