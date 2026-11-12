#!/usr/bin/env python3
"""Monitor the MACELES-LES density scan (6 molecules x 10 densities = 60 states).

Usage:
    python scripts/monitor.py               # one report
    python scripts/monitor.py --watch 900   # repeat every 900 s
"""
from __future__ import annotations
import argparse, csv, json, math, shutil, subprocess, time
from pathlib import Path

SCAN = Path("/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan")
MOL = ("H2O", "NH3", "CH4", "C2H4", "C2H6", "CO2")
N_ATOMS = {"H2O": 2001, "NH3": 2000, "CH4": 2500, "C2H4": 3000, "C2H6": 4000, "CO2": 2001}
CORES = {"H2O": 32, "NH3": 32, "CH4": 32, "C2H4": 48, "C2H6": 48, "CO2": 32}

# measured CO2 (2001 atoms, 32 cores) equil rate h/ns at 4 densities
_RHO_PT = [0.6700, 0.4533, 0.2367, 0.0922]
_H_PT = [37.33, 39.95, 49.26, 87.67]


def rate_co2(rhostar: float) -> float:
    """log-log interpolate/extrapolate CO2 h/ns from 4 measured points."""
    lr = [math.log(x) for x in _RHO_PT]
    lh = [math.log(x) for x in _H_PT]
    lx = math.log(max(rhostar, 1e-6))
    for i in range(len(lr) - 1):
        if lr[i] >= lx >= lr[i + 1]:
            t = (lx - lr[i]) / (lr[i + 1] - lr[i])
            return math.exp(lh[i] + t * (lh[i + 1] - lh[i]))
    if lx > lr[0]:   # denser than 0.67
        t = (lx - lr[0]) / (lr[1] - lr[0])
        return math.exp(lh[0] + t * (lh[1] - lh[0]))
    t = (lx - lr[-2]) / (lr[-1] - lr[-2])   # sparser than 0.0922
    return math.exp(lh[-2] + t * (lh[-1] - lh[-2]))


def scale(mol: str) -> float:
    return (N_ATOMS[mol] / 2001.0) / math.sqrt(CORES[mol] / 32.0)


def _secs(x: str) -> int:
    try:
        s = 0
        if "-" in x:
            d, x = x.split("-", 1)
            s += int(d) * 86400
        a = x.split(":")
        if len(a) == 3:
            s += int(a[0]) * 3600 + int(a[1]) * 60 + int(a[2])
        elif len(a) == 2:
            s += int(a[0]) * 60 + int(a[1])
        return s
    except Exception:  # noqa: BLE001
        return 0


def _shard_of(name: str) -> str:
    if name.startswith("cdft_s"):
        return "shard_" + name[len("cdft_s"):]
    return "shard_" + name[len("cdft_"):]


def squeue_lines():
    try:
        out = subprocess.run(
            ["squeue", "-u", "m9s005258", "-o", "%.10i %.14j %.2t %.8M %.12L %R",
             "--noheader"], capture_output=True, text=True, timeout=20).stdout
    except Exception as e:  # noqa: BLE001
        return [f"(squeue error {e})"]
    return [l for l in out.splitlines() if "cdft" in l]


def collect_sim_ids():
    ids = set()
    for pat in ("shard_*/attempt-01/manifest.csv", "shard_*_*/attempt-01/manifest.csv"):
        for man in sorted(SCAN.glob(pat)):
            try:
                for r in csv.DictReader(open(man)):
                    ids.add(r["sim_id"])
            except Exception:  # noqa: BLE001
                pass
    return sorted(ids)


def report() -> str:
    lines = [f"== {time.strftime('%Y-%m-%d %H:%M:%S')} =="]
    sl = squeue_lines()
    lines.append(f"  running cdft jobs: {len(sl)}")
    for l in sl:
        lines.append(f"    {l}")

    ids = collect_sim_ids()
    per = {m: [0, 0] for m in MOL}
    total_md = total_post = 0
    for sid in ids:
        mol = sid.split("-")[0]
        if mol not in per:
            continue
        per[mol][1] += 1
        if (SCAN / "traj" / sid / "meta.json").exists():
            per[mol][0] += 1
            total_md += 1
        if (SCAN / "post" / sid / "les_c1c2.json").exists():
            total_post += 1
    lines.append("  states(MD done/total): " +
                 "  ".join(f"{m}:{per[m][0]}/{per[m][1]}" for m in MOL))
    lines.append(f"  totals: {len(ids)} states, MD done={total_md}, post done={total_post}")

    hs = []
    for meta in sorted((SCAN / "traj").glob("*/meta.json")):
        try:
            h = json.loads(meta.read_text()).get("timing", {}).get("hours_per_ns")
            if h:
                hs.append(h)
        except Exception:  # noqa: BLE001
            pass
    if hs:
        lines.append(f"  timing: {len(hs)} states, mean {sum(hs)/len(hs):.1f} h/ns, "
                     f"min {min(hs):.1f}, max {max(hs):.1f}")

    traj = list((SCAN / "traj").glob("*/run/traj.dump"))
    sz = sum(p.stat().st_size for p in traj)
    free = shutil.disk_usage(SCAN).free
    lines.append(f"  traj.dump: {len(traj)} files, {sz/1e9:.2f} GB; free {free/1e9:.0f} GB")

    # workload-vs-timeleft risk (wall-clock days remaining to finish each job)
    warns = []
    for l in sl:
        p = l.split()
        if len(p) < 6 or p[2] != "R":
            continue
        name = p[1]
        left = _secs(p[4])
        if left <= 0:
            continue
        man = SCAN / _shard_of(name) / "attempt-01" / "manifest.csv"
        if not man.exists():
            continue
        rem_h = 0.0
        mol = name.split("_")[1] if name.startswith("cdft_") and name[5] != "s" else "CO2"
        # job names cdft_<mol>_<k>; cdft_s<k> -> CO2
        if name.startswith("cdft_s"):
            mol = "CO2"
        else:
            parts = name.split("_")
            mol = parts[1] if len(parts) > 1 else "CO2"
        try:
            for r in csv.DictReader(open(man)):
                sid = r["sim_id"]
                if (SCAN / "traj" / sid / "meta.json").exists():
                    continue
                rhostar = float(sid.rsplit("rho", 1)[-1])
                rem_h += 1.1 * rate_co2(rhostar) * scale(mol)
        except Exception:  # noqa: BLE001
            continue
        if rem_h / 24.0 > left / 86400.0 * 0.95:
            warns.append(f"{name}: need ~{rem_h/24:.1f}d > left {left/86400:.1f}d")
    if warns:
        lines.append("  RISK(timeout): " + "; ".join(warns))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=float, default=0)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    log = Path(args.log) if args.log else SCAN / "logs" / "monitor.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        block = report()
        print(block, flush=True)
        with open(log, "a") as fh:
            fh.write(block + "\n")
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
