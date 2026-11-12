#!/usr/bin/env python3
"""Split a density-scan manifest into N balanced shard manifests.

Each shard gets a contiguous block of rows from ``density_scan.csv`` and is
written to ``<root>/shard_<i>/<attempt>/manifest.csv``.  With
``--remote-prefix`` the data paths are rewritten from the local root to the
scheduler-visible vePFS prefix, so the same manifest can be submitted as a
remote task.

    python -m sampling.split_scan --root bench/maceles_density_scan \
        --shards 4 --attempt attempt-01 \
        --remote-prefix /vepfs-.../workspace-state/shared/bza/symmetrix-polar/bench/maceles_density_scan
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True,
                    help="scan root holding density_scan.csv")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--attempt", default="attempt-01")
    ap.add_argument("--remote-prefix", default=None,
                    help="rewrite local data paths to this prefix")
    ap.add_argument("--dump-every", type=int, default=None,
                    help="override the dump cadence (steps) in every row")
    args = ap.parse_args(argv)

    root = Path(args.root)
    rows = list(csv.DictReader(open(root / "density_scan.csv")))
    if not rows:
        raise SystemExit(f"no rows in {root/'density_scan.csv'}")
    if args.dump_every:
        for r in rows:
            r["dump_every"] = str(args.dump_every)
    n = len(rows)
    per = (n + args.shards - 1) // args.shards
    fields = list(rows[0].keys())
    local_prefix = str(root.resolve())
    for s in range(args.shards):
        chunk = rows[s * per:(s + 1) * per]
        if not chunk:
            continue
        if args.remote_prefix:
            for r in chunk:
                r["data"] = r["data"].replace(local_prefix,
                                              args.remote_prefix.rstrip("/"))
        out = root / f"shard_{s + 1}" / args.attempt / "manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(chunk)
        print(f"shard_{s + 1}: {len(chunk)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
