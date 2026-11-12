#!/usr/bin/env python3
"""Batch MACELES/LES c1/c2 sampling with permanent trajectory retention.

Each manifest row is one simulation state.  For every row this driver creates

    <outroot>/<sim_id>/run/traj.dump    # LAMMPS dump, kept forever
    <outroot>/<sim_id>/run/log.les_md
    <outroot>/<sim_id>/f1/, f2/         # PRL F1/F2 (via cdft_dataset.process_bulk)
    <outroot>/<sim_id>/meta.json        # model/cubin/data hashes + parameters
    <outroot>/<sim_id>/les_c1c2.json    # extraction manifest

and appends one line per run to <outroot>/summary.csv.  Existing sim_id
directories are never overwritten: the run stops with an error, unless
--resume is given and the simulation already has les_c1c2.json (then it is
skipped).  Trajectories are never deleted.

Manifest columns (header required):
    sim_id, data, traj, species, temperature, timestep, steps, dump_every,
    density, c2_pair, seed, elements

- --run:           use `data`; write/execute the LAMMPS input; then analyze.
- --analyze-only: use `traj`; skip MD and only extract c1/c2.
- density/c2_pair/seed/elements may be empty; density is derived from the
  data/dump, c2_pair defaults to "O O", elements to "C H N O".
- sim_id may be empty; the default is <species>-r<density>-<T>K-s<seed>.

Examples
--------
Generate the six-species 3000-atom manifest (100 ps production, 0.5 ps dump):

    python sampling/les_batch.py --template six \
        --manifest six_3000_100ps.csv --steps 200000 --dump-every 1000

Run it (trajectories under ./maceles_samples/<sim_id>/run/traj.dump):

    python sampling/les_batch.py --manifest six_3000_100ps.csv \
        --outroot maceles_samples --run --resume

Reuse an existing 100 ps dump without rerunning MD:

    python sampling/les_batch.py --manifest manifest.csv \
        --outroot maceles_samples --analyze-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from sampling.les_c1c2 import (  # noqa: E402
    DEFAULT_CUBIN, DEFAULT_HOST_ARTIFACT, DEFAULT_LMP, DEFAULT_MODEL,
    analyze_trajectory, parse_elements, parse_pair, run_md,
)

SIX_ROOT = Path(
    "/workspace/group/shared/bza/symmetrix-polar/bench/q4_maceles_off/six_3000")
SIX = {
    "H2O": ("0.0334", "O O"),
    "NH3": ("0.0241", "N N"),
    "CH4": ("0.0158", "C C"),
    "C2H4": ("0.0123", "C C"),
    "C2H6": ("0.0105", "C C"),
    "CO2": ("0.0137", "C O"),
}
COLUMNS = ["sim_id", "data", "traj", "species", "temperature", "timestep",
           "steps", "dump_every", "density", "c2_pair", "seed", "elements"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_template(path: Path, steps: int, dump_every: int,
                   temperature: float, timestep: float) -> None:
    rows = []
    for species, (rho, pair) in SIX.items():
        rows.append({
            "sim_id": f"{species}-r{rho}-{temperature:g}K-s0",
            "data": str(SIX_ROOT / species / "data.lammps"),
            "traj": "",
            "species": species,
            "temperature": temperature,
            "timestep": timestep,
            "steps": steps,
            "dump_every": dump_every,
            "density": rho,
            "c2_pair": pair,
            "seed": 0,
            "elements": "C H N O",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"[batch] wrote {len(rows)} rows to {path}")


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = set(COLUMNS) - set(rows[0]) if rows else set()
    if missing:
        raise SystemExit(f"manifest is missing columns: {sorted(missing)}")
    return rows


def append_summary(outroot: Path, record: dict) -> None:
    path = outroot / "summary.csv"
    fields = ["sim_id", "species", "temperature", "density", "steps",
              "dump_every", "timestep", "traj", "outdir", "n_frames",
              "us_per_atom_step", "hours_per_ns", "status"]
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in fields})


def run_row(row: dict, args, outroot: Path) -> dict:
    species = row["species"].strip()
    elements = parse_elements(row["elements"] or args.elements)
    pair = parse_pair(row["c2_pair"] or args.c2_pair or "O O")
    temperature = float(row["temperature"] or args.temperature)
    timestep = float(row["timestep"] or args.timestep)
    steps = int(row["steps"] or args.steps)
    dump_every = int(row["dump_every"] or args.dump_every)
    density = float(row["density"]) if row["density"].strip() else None
    seed_raw = row["seed"].strip()
    seed = int(seed_raw) if seed_raw and int(seed_raw) > 0 else int(args.seed)
    sim_id = row["sim_id"].strip() or (
        f"{species}-r{density if density is not None else 'auto'}-"
        f"{temperature:g}K-s{seed}")
    root = outroot / sim_id
    done = (root / "les_c1c2.json").exists()
    md_done = (root / "meta.json").exists() and not done
    if root.exists():
        if args.resume and done:
            print(f"[batch] skip {sim_id} (already complete)")
            return {"sim_id": sim_id, "status": "skipped"}
        if args.resume and md_done and getattr(args, "no_analyze", False):
            print(f"[batch] skip {sim_id} (MD complete, no-analyze mode)")
            return {"sim_id": sim_id, "status": "skipped"}
        if args.resume:
            import time as _time
            keep = root.with_name(root.name + f".partial.{int(_time.time())}")
            root.rename(keep)
            print(f"[batch] partial {sim_id} kept at {keep}; restarting")
        else:
            raise SystemExit(
                f"refusing to overwrite existing {root}; use --resume to skip "
                f"completed rows or choose another --outroot")
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "sim_id": sim_id,
        "species": species,
        "elements": list(elements),
        "temperature_K": temperature,
        "timestep_ps": timestep,
        "steps": steps,
        "dump_every": dump_every,
        "density_mol_A3": density,
        "c2_pair": list(pair),
        "seed": seed,
        "model": str(Path(args.model).resolve()),
        "model_sha256": sha256(args.model),
    }
    if args.cubin and Path(args.cubin).exists():
        meta["cubin"] = str(Path(args.cubin).resolve())
        meta["cubin_sha256"] = sha256(args.cubin)
    if args.host_artifact and Path(args.host_artifact).exists():
        meta["host_artifact"] = str(Path(args.host_artifact).resolve())
        meta["host_artifact_sha256"] = sha256(args.host_artifact)
    if args.run:
        if not row["data"].strip():
            raise SystemExit(f"row {sim_id}: --run requires data")
        md = run_md(
            data=row["data"], species=species, elements=elements,
            temperature=temperature, timestep=timestep, steps=steps,
            dump_every=dump_every, workdir=root / "run",
            equil_steps=args.equil_steps, seed=seed,
            model=args.model, cubin=args.cubin, lmp=args.lmp,
            density=density, gpus=args.gpus, omp_threads=args.omp_threads,
            np_ranks=args.np_ranks, mpirun=args.mpirun,
            host_artifact=args.host_artifact,
            ld_library_path=args.ld_library_path)
        traj = md["dump"]
        density = md["density_mol_A3"]
        meta.update({
            "data": md["data"], "data_sha256": md["data_sha256"],
            "traj": traj, "traj_sha256": md["dump_sha256"],
            "timing": md["timing"],
        })
        if getattr(args, "no_analyze", False):
            # The workspace sits under a per-scope quota; 60 raw 1 ns dumps at
            # 25 fs need ~250 GB.  Compress each finished trajectory (~4-5x on
            # text dumps) right after MD so the steady-state footprint is small.
            if not getattr(args, "no_compress_traj", False):
                import gzip
                import shutil as _shutil
                traj_path = Path(traj)
                if traj_path.exists() and traj_path.suffix != ".gz":
                    gz = traj_path.with_name(traj_path.name + ".gz")
                    with open(traj_path, "rb") as fi, \
                            gzip.open(gz, "wb", compresslevel=5) as fo:
                        _shutil.copyfileobj(fi, fo, 8 << 20)
                    raw_bytes = traj_path.stat().st_size
                    gz_sha = sha256(gz)
                    traj_path.unlink()
                    traj = str(gz)
                    md["dump"] = traj
                    md["dump_sha256"] = gz_sha
                    meta.update({"traj": traj, "traj_sha256": gz_sha,
                                 "traj_raw_bytes": raw_bytes,
                                 "traj_gz_bytes": gz.stat().st_size})
                    print(f"[batch] {sim_id}: compressed traj "
                          f"{raw_bytes/1e9:.2f} -> "
                          f"{gz.stat().st_size/1e9:.2f} GB")
            meta["status"] = "md_complete"
            (root / "meta.json").write_text(
                json.dumps(meta, indent=2, default=str))
            record = {
                "sim_id": sim_id, "species": species,
                "temperature": temperature, "density": density,
                "steps": steps, "dump_every": dump_every,
                "timestep": timestep, "traj": meta.get("traj", ""),
                "outdir": str(root), "n_frames": "",
                "us_per_atom_step": meta.get("timing", {}).get(
                    "us_per_atom_step", ""),
                "hours_per_ns": meta.get("timing", {}).get(
                    "hours_per_ns", ""),
                "status": "md_complete",
            }
            append_summary(outroot, record)
            print(f"[batch] {sim_id}: MD done, traj={meta.get('traj')}")
            return record
        rec, density = analyze_trajectory(
            traj, root, species, elements, temperature, density=density,
            pair=pair, stride=args.stride, c2_bins=args.c2_bins,
            bins=args.bins)
        meta["density_mol_A3"] = density
    else:
        if not row["traj"].strip():
            raise SystemExit(f"row {sim_id}: --analyze-only requires traj")
        traj = row["traj"].strip()
        if not Path(traj).exists():
            raise SystemExit(f"row {sim_id}: trajectory not found: {traj}")
        meta.update({"traj": traj, "traj_sha256": sha256(traj)})
        rec, density = analyze_trajectory(
            traj, root, species, elements, temperature, density=density,
            pair=pair, stride=args.stride, c2_bins=args.c2_bins,
            bins=args.bins)
        meta["density_mol_A3"] = density
    meta["status"] = "complete"
    (root / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    record = {
        "sim_id": sim_id, "species": species, "temperature": temperature,
        "density": density, "steps": steps, "dump_every": dump_every,
        "timestep": timestep, "traj": meta.get("traj", ""),
        "outdir": str(root), "n_frames": rec.get("n_frames", ""),
        "us_per_atom_step": meta.get("timing", {}).get(
            "us_per_atom_step", ""),
        "hours_per_ns": meta.get("timing", {}).get("hours_per_ns", ""),
        "status": "complete",
    }
    append_summary(outroot, record)
    print(f"[batch] {sim_id}: F1/F2 -> {root}, traj={meta.get('traj')}")
    return record


WORKSPACE_OWNER_UID = 166536


def drop_to_workspace_owner(outroot=None) -> None:
    """Run as the workspace owner so outputs land in the right quota.

    On root-squashed vePFS nodes a root process writes files owned by
    `nobody` (uid 65534); those bytes hit a small per-identity quota and the
    dump dies with EDQUOT after a few GB.  Running as the workspace owner
    (uid 166536) charges the writes to the real workspace quota instead.
    """
    if os.getuid() != 0:
        print(f"[batch] running as uid {os.getuid()}, not dropping privileges")
        return
    # Existing output directories may be owned by another uid (e.g. an
    # earlier attempt that ran unsquashed); hand them to 166536 before the
    # uid switch, otherwise the drop has no write permission.
    if outroot:
        import subprocess
        target = Path(outroot)
        while target != target.parent and not target.exists():
            target = target.parent
        for path in [target] + sorted(target.rglob("*")):
            try:
                os.chown(path, WORKSPACE_OWNER_UID, WORKSPACE_OWNER_UID)
            except OSError:
                pass
        subprocess.run(["chmod", "-R", "u+rwX,g+rwX", str(target)],
                       check=False, capture_output=True)
    try:
        os.setgroups([])
        os.setgid(WORKSPACE_OWNER_UID)
        os.setuid(WORKSPACE_OWNER_UID)
        print(f"[batch] dropped privileges to uid={os.getuid()} "
              f"gid={os.getgid()}")
    except OSError as exc:
        print(f"[batch] WARNING: could not drop privileges: {exc}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Batch MACELES/LES c1/c2 sampling with trajectory retention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV manifest path")
    ap.add_argument("--outroot", default=None,
                    help="root for <sim_id>/ outputs (trajectories kept)")
    ap.add_argument("--template", choices=["six"], default=None,
                    help="write a template manifest instead of running")
    ap.add_argument("--run", action="store_true", help="run MD per row")
    ap.add_argument("--analyze-only", action="store_true",
                    help="analyze existing trajectories per row")
    ap.add_argument("--resume", action="store_true",
                    help="skip completed sim_ids; keep partial runs aside")
    ap.add_argument("--no-analyze", action="store_true",
                    help="run MD only and keep the trajectory; extract "
                         "c1/c2 later with --analyze-only")
    ap.add_argument("--no-compress-traj", action="store_true",
                    help="keep traj.dump uncompressed after MD (default: gzip "
                         "it to stay inside the workspace quota)")
    ap.add_argument("--drop-priv", action="store_true",
                    help="switch to uid 166536 when started as root (off by "
                         "default: the workspace dirs belong to other uids)")
    ap.add_argument("--steps", type=int, default=2000000)
    ap.add_argument("--equil-steps", type=int, default=200000,
                    help="NVT equilibration steps before production/dump (default 200000 = 100 ps at 0.5 fs)")
    ap.add_argument("--seed", type=int, default=12345,
                    help="velocity seed for rows without a seed")
    ap.add_argument("--dump-every", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--timestep", type=float, default=0.0005)
    ap.add_argument("--elements", nargs="+", default=["C", "H", "N", "O"])
    ap.add_argument("--c2-pair", nargs="+", default=["O", "O"])
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--c2-bins", type=int, default=80)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=0,
                    help="GPUs per rank; 0 = CPU")
    ap.add_argument("--np-ranks", type=int, default=1,
                    help="MPI ranks for CPU runs (mpirun -np N)")
    ap.add_argument("--mpirun", default=None,
                    help="mpirun binary (default: mpirun on PATH)")
    ap.add_argument("--omp-threads", default="1")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--cubin", default=str(DEFAULT_CUBIN))
    ap.add_argument("--host-artifact", default=str(DEFAULT_HOST_ARTIFACT))
    ap.add_argument("--lmp", default=str(DEFAULT_LMP))
    ap.add_argument("--ld-library-path", default=None)
    args = ap.parse_args(argv)
    if args.run and args.analyze_only:
        ap.error("choose one of --run / --analyze-only")

    if (args.run or args.analyze_only) and getattr(args, "drop_priv", False):
        drop_to_workspace_owner(args.outroot)

    manifest = Path(args.manifest)
    if args.template:
        write_template(manifest, args.steps, args.dump_every,
                       args.temperature, args.timestep)
        if not (args.run or args.analyze_only):
            return 0
    if not (args.run or args.analyze_only):
        ap.error("nothing to do: pass --run or --analyze-only (or --template)")
    if not args.outroot:
        ap.error("--outroot is required unless --template is used")
    rows = read_manifest(manifest)
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    if args.ld_library_path is not None:
        # les_c1c2 reads the default unless overridden per call
        args.ld_library_path = args.ld_library_path
    else:
        from sampling.les_c1c2 import DEFAULT_LD_LIBRARY_PATH
        args.ld_library_path = DEFAULT_LD_LIBRARY_PATH
    for i, row in enumerate(rows, 1):
        print(f"[batch] {i}/{len(rows)} {row.get('sim_id') or row.get('species')}")
        run_row(row, args, outroot)
    print(f"[batch] done: {len(rows)} rows, trajectories under {outroot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
