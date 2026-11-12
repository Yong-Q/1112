#!/usr/bin/env python3
"""Compact status of a running MACELES density scan.

    python -m sampling.scan_status --root <scan_root> --watch 300

Writes one block per pass to stdout (and appends to <root>/analysis_scan/status.log):
per-shard step/total, temperature, measured rate and ETA, how many states are
finished, how much trajectory data is on disk, and whether the post-processing
loops are alive.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

STEP_RE = re.compile(r"^\s+(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s*$", re.M)
EQUIL_STEPS = 200_000
PROD_STEPS = 2_000_000


def shard_state(shard: Path):
    runs = sorted(shard.glob("attempt-*/samples/*/run"))
    if not runs:
        return None
    run = max(runs, key=lambda p: p.stat().st_mtime)
    out = run / "stdout.log"
    if not out.exists():
        return None
    m = STEP_RE.findall(out.read_text(errors="ignore"))
    if not m:
        return None
    step, temp, pe = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
    meta = run.parent / "meta.json"
    traj = run / "traj.dump"
    producing = traj.exists() and traj.stat().st_size > 0
    total = PROD_STEPS if producing else EQUIL_STEPS
    return {"sim": run.parent.name, "step": step, "temp": temp, "pe": pe,
            "phase": "prod" if producing else "equil", "total": total,
            "done": meta.exists(), "mtime": out.stat().st_mtime,
            "traj_mb": (traj.stat().st_size / 1e6) if producing else 0.0}


def loops_alive():
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return 0
    return sum(out.count(f"sampling.{m}") for m in
               ("scan_watch", "zenodo_upload", "zenodo_batch_traj"))


def report(root: Path) -> str:
    lines = [f"== {time.strftime('%Y-%m-%d %H:%M:%S')} =="]
    total_done = 0
    for shard in sorted(root.glob("shard_*")):
        st = shard_state(shard)
        if not st:
            continue
        frac = 100.0 * st["step"] / st["total"]
        rate = ""
        age = max(1.0, time.time() - st["mtime"])
        lines.append(f"  {shard.name}: {st['sim'][:30]:30s} "
                     f"[{st['phase']:5s}] {st['step']:>8,}/{st['total']:,} "
                     f"({frac:5.1f}%)  T={st['temp']:7.1f} K  "
                     f"traj={st['traj_mb']:6.1f} MB  "
                     f"{'DONE' if st['done'] else ''}")
    done = len(list(root.glob("shard_*/attempt-*/samples/*/meta.json")))
    traj = list(root.glob("shard_*/attempt-*/samples/*/run/traj.dump*"))
    traj += list(root.glob("traj_archive/*"))
    size = sum(p.stat().st_size for p in traj if p.is_file())
    free = shutil.disk_usage(root).free
    zen = ""
    zpath = root / ".zenodo_traj.json"
    if zpath.exists():
        try:
            d = json.loads(zpath.read_text())
            zen = f"  zenodo_traj={len(d.get('done', {}))}"
        except Exception:
            pass
    lines.append(f"  finished states: {done}/60   trajectories on disk: "
                 f"{len(traj)} ({size/1e9:.1f} GB)   free: {free/1e9:.0f} GB{zen}")
    lines.append(f"  post-processing loops alive: {loops_alive()}/3")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--watch", type=float, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    log = Path(args.log) if args.log else root / "analysis_scan" / "status.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        block = report(root)
        print(block, flush=True)
        with open(log, "a") as fh:
            fh.write(block + "\n")
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
