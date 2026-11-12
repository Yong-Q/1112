#!/usr/bin/env python3
"""Upload finished trajectories to Zenodo in bounded batches.

Keeps the local disk small without hammering a slow link: each pass uploads at
most ``--batch`` trajectories, deletes a local file only after Zenodo returns
2xx for its final commit, and records what was uploaded so a rerun never
re-sends the same file.

    export ZENODO_TOKEN=...
    python -m sampling.zenodo_batch_traj --root bench/maceles_density_scan \
        --record-id 23901522 --batch 3 --watch 3600

One file per pass is uploaded as ``trajectories/<sim_id>.dump.gz``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from sampling.zenodo_upload import create_draft, new_version, upload


def finished(root: Path, done: dict, archive_dir=None):
    """Return [(sim_id, traj)] for finished states not yet uploaded.

    The archive step drops the worker copy as soon as the archived copy is
    complete, so a failed upload must be retried from ``archive_dir``.
    """
    metas = list(root.glob("shard_*/attempt-*/samples/*/meta.json"))
    metas += list(root.glob("samples/*/meta.json"))
    out = []
    for meta in sorted(metas):
        state = meta.parent
        if "samples_old" in str(state) or "_legacy" in str(state):
            continue
        sid = state.name
        if sid in done:
            continue
        traj = state / "run" / "traj.dump.gz"
        if not traj.exists():
            traj = state / "run" / "traj.dump"
        if not traj.exists() and archive_dir:
            cand = Path(archive_dir) / f"{sid}.dump.gz"
            if cand.exists():
                traj = cand
        if traj.exists():
            out.append((sid, traj))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--record-id", required=True)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--watch", type=float, default=0)
    ap.add_argument("--max-bytes", type=float, default=45e9,
                    help="roll over to a new Zenodo record before this size "
                         "(Zenodo's limit is 50 GB per record)")
    ap.add_argument("--delete-after", action="store_true",
                    help="remove the local file once Zenodo confirmed it "
                         "(default: keep it, space is not the constraint)")
    ap.add_argument("--state", default=None)
    ap.add_argument("--archive-dir", default=None,
                    help="copy finished trajectories here (agent-owned) and "
                         "drop the worker-owned original, so the worker "
                         "identity quota stays small while a local copy remains")
    args = ap.parse_args(argv)

    root = Path(args.root)
    state_file = Path(args.state) if args.state else root / ".zenodo_traj.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    done = state.setdefault("done", {})
    recs = state.setdefault("records", [{"id": args.record_id, "bytes": 0}])

    def save():
        state_file.write_text(json.dumps(state, indent=2))

    while True:
        try:
            todo = finished(root, done, args.archive_dir)
        except Exception as exc:                 # noqa: BLE001
            print(f"[traj] scan error: {type(exc).__name__}: {exc}")
            time.sleep(max(60, args.watch))
            continue
        todo = todo[:max(1, args.batch)]
        print(f"[traj {time.strftime('%H:%M:%S')}] {len(todo)} to upload "
              f"({len(done)} done, {len(recs)} record(s), "
              f"{recs[-1]['bytes']/1e9:.1f} GB in the last)", flush=True)
        archive = Path(args.archive_dir) if args.archive_dir else None
        if archive:
            archive.mkdir(parents=True, exist_ok=True)
        for sid, traj in todo:
            traj = Path(traj)
            # Never treat an already-archived file as a new source: doing so
            # would unlink the only copy (dst == traj).
            if archive and traj.parent.resolve() != archive.resolve():
                import shutil
                suffix = ".dump.gz" if traj.name.endswith(".dump.gz") else ".dump"
                dst = archive / f"{sid}{suffix}"
                if not dst.exists():
                    shutil.copy2(traj, dst)
                    print(f"[traj] archived {dst.name} "
                          f"({dst.stat().st_size/1e6:.0f} MB)", flush=True)
                if dst.stat().st_size == traj.stat().st_size:
                    traj.unlink()
                traj = dst
            key = f"trajectories/{sid}.dump.gz" if traj.suffix == ".gz" \
                else f"trajectories/{sid}.dump"
            size = traj.stat().st_size
            if recs[-1]["bytes"] + size > args.max_bytes:
                try:
                    rec = new_version(recs[-1]["id"])
                except Exception as exc:         # noqa: BLE001
                    print(f"[traj] versioning failed ({exc}); new record")
                    rec = create_draft("trajectories")
                recs.append({"id": rec["id"], "bytes": 0})
                print(f"[traj] continued in record {rec['id']} "
                      f"(version {len(recs)})", flush=True)
            ok = upload(recs[-1]["id"], key, traj)
            if ok:
                if args.delete_after:
                    traj.unlink()
                recs[-1]["bytes"] += size
                done[sid] = {"key": key, "record": recs[-1]["id"], "bytes": size}
                save()
                print(f"[traj] uploaded {traj}" +
                      (" and removed" if args.delete_after else ""), flush=True)
            else:
                print(f"[traj] keeping {traj} (upload failed)", flush=True)
        save()
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
