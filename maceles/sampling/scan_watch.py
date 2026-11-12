#!/usr/bin/env python3
"""Watch a running scan: analyse each finished state, then delete its trajectory.

    python -m sampling.scan_watch --root bench/maceles_density_scan --interval 600

Every ``--interval`` seconds it looks for ``samples/<sim_id>/meta.json`` states
that have no ``analysis.json`` yet, runs the full post-processing (c1/c2, all
pair g(r) + anisotropy, structure) and — with ``--prune`` — removes
``run/traj.dump*`` afterwards.  Finished runs are aggregated into
``<out>/scan_summary.csv`` and the figure set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sampling.scan_analysis import aggregate, discover_states, run_states


def pending(root: Path, out: Path):
    todo = []
    for sid, sp, temp, rho, state, traj in discover_states(root):
        # only states whose MD has finished: meta.json is written after the
        # trajectory is gzipped and closed
        if not (state / "meta.json").exists():
            continue
        if not (out / sid / "analysis.json").exists():
            todo.append((sid, state, traj))
    return todo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--interval", type=float, default=600)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--prune", action="store_true", default=True)
    ap.add_argument("--keep-traj", action="store_true")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--c2-stride", type=int, default=10)
    ap.add_argument("--pair-frames", type=int, default=20)
    ap.add_argument("--zenodo-id", help="Zenodo deposition id for trajectories")
    ap.add_argument("--zenodo-upload-traj", action="store_true",
                    help="upload each trajectory to Zenodo, then delete it")
    args = ap.parse_args(argv)

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "analysis_scan"
    out.mkdir(parents=True, exist_ok=True)
    prune = args.prune and not args.keep_traj

    while True:
        todo = pending(root, out)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if todo:
            print(f"[watch {stamp}] {len(todo)} state(s) to process", flush=True)
            try:
                run_states(root, out, n_frames=args.frames,
                           c2_stride=args.c2_stride,
                           pairs=True, pair_frames=args.pair_frames,
                           prune_traj=prune and not args.zenodo_upload_traj,
                           require_meta=True)
                aggregate(out)
            except Exception as exc:             # noqa: BLE001
                print(f"[watch] pass failed: {type(exc).__name__}: {exc}",
                      flush=True)
        else:
            print(f"[watch {stamp}] nothing new", flush=True)
        if args.zenodo_id and args.zenodo_upload_traj and todo:
            import os
            import subprocess
            import sys
            for sid, state, traj in todo:
                tr = Path(traj)
                if not tr.exists():
                    continue
                key = f"trajectories/{sid}.dump.gz" if tr.suffix == ".gz" \
                    else f"trajectories/{sid}.dump"
                env = dict(os.environ)
                env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
                rc = subprocess.run(
                    [sys.executable, "-m", "sampling.zenodo_upload",
                     "--root", str(state), "--deposition-id", args.zenodo_id,
                     "--single-file", str(tr), "--key", key, "--delete-after"],
                    env=env)
                print(f"[watch] zenodo upload {key}: rc={rc.returncode}",
                      flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
