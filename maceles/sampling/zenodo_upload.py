#!/usr/bin/env python3
"""Mirror scan output to Zenodo (new InvenioRDM records API).

Zenodo replaced the legacy ``/api/depositions`` API; this client uses the
current one:

    POST   /api/records                       create a draft
    POST   /api/records/{id}/draft/files      initialise a file
    PUT    /api/records/{id}/draft/files/{k}/content
    POST   /api/records/{id}/draft/files/{k}/commit
    POST   /api/records/{id}/draft/actions/publish

A draft stays editable until published: files can be added, replaced (same key)
or deleted as often as needed; publishing mints the DOI and freezes the record
(later changes need a new version).

Two records keep raw trajectories and post-processing separate::

    export ZENODO_TOKEN=...
    python -m sampling.zenodo_upload --root .../analysis_scan --kind analysis \
        --bundle-states --watch 1800
    python -m sampling.zenodo_upload --single-file traj.dump.gz \
        --key trajectories/<sim>.dump.gz --record-id <traj-record> --delete-after

Without ``--record-id`` the first run creates a draft, prints its id and stores
it in ``<root>/.zenodo_record.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = os.environ.get("ZENODO_API", "https://zenodo.org/api").rstrip("/")

BASE_META = {
    "publication_date": "2026-11-03",
    "resource_type": {"id": "dataset"},
    "license": {"id": "cc-by-4.0"},
    "creators": [{"person_or_org": {"type": "personal", "name": "Qiu, Yong"}}],
    "keywords": ["cDFT", "MACELES", "latent Ewald summation", "c1/c2", "g(r)"],
}

PRESETS = {
    "analysis": {
        "title": "MACELES PRL density scan — c1/c2 and pair correlations",
        "description": (
            "Post-processing of the MACELES (MACE + Latent Ewald Summation) "
            "bulk density scan: c1/c2 (rho, Vext, mu, r, c2, c2k), all-pair "
            "g(r) with spherical-harmonic anisotropy, intramolecular bond "
            "lengths/angles/dihedrals, molecular axes with rotational "
            "correlation, centre-of-mass trajectories and MSD, per state."
        ),
    },
    "trajectories": {
        "title": "MACELES PRL density scan — raw trajectories",
        "description": (
            "Raw LAMMPS trajectories (dump: id mol type x y z fx fy fz) of the "
            "MACELES bulk density scan, 0.5 ps frames, one file per state."
        ),
    },
}


def _headers():
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        path = Path("/workspace/group/shared/bza/.secrets/zenodo_token")
        if path.exists():
            token = path.read_text().strip()
    if not token:
        raise SystemExit("ZENODO_TOKEN is not set")
    return {"Authorization": f"Bearer {token}"}


def _meta(kind, overrides=None):
    meta = dict(BASE_META)
    meta.update(PRESETS.get(kind, PRESETS["analysis"]))
    if overrides:
        meta.update(overrides)
    return meta


def create_draft(kind, overrides=None):
    r = requests.post(f"{API}/records",
                      json={"metadata": _meta(kind, overrides),
                            "files": {"enabled": True}},
                      headers=_headers(), timeout=60)
    r.raise_for_status()
    rec = r.json()
    print(f"[zenodo] created draft {rec['id']} ({rec['metadata']['title']})")
    return rec


def get_draft(record_id):
    r = requests.get(f"{API}/records/{record_id}/draft", headers=_headers(),
                     timeout=60)
    r.raise_for_status()
    return r.json()


def list_files(record_id):
    r = requests.get(f"{API}/records/{record_id}/draft/files",
                     headers=_headers(), timeout=60)
    if r.status_code >= 400:
        return {}
    return {e["key"]: e.get("size", 0) for e in r.json().get("entries", [])}


def delete_file(record_id, key):
    requests.delete(f"{API}/records/{record_id}/draft/files/{key}",
                    headers=_headers(), timeout=60)


def upload(record_id, key, path: Path, attempts: int = 3) -> bool:
    """Upload one file, retrying transient network errors with backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return _upload(record_id, key, path)
        except Exception as exc:                 # noqa: BLE001
            print(f"[zenodo] upload error {key} (try {attempt}/{attempts}): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(15 * attempt)
    return False


def _upload(record_id, key, path: Path) -> bool:
    key = key.replace(os.sep, "/")
    if key in list_files(record_id):
        delete_file(record_id, key)
    r = requests.post(f"{API}/records/{record_id}/draft/files",
                      json=[{"key": key}], headers=_headers(), timeout=60)
    if r.status_code >= 400:
        print(f"[zenodo] init failed {key}: {r.status_code} {r.text[:200]}")
        return False
    with open(path, "rb") as fh:
        r = requests.put(f"{API}/records/{record_id}/draft/files/{key}/content",
                         data=fh,
                         headers={**_headers(),
                                  "Content-Type": "application/octet-stream"},
                         timeout=3600)
    if r.status_code >= 400:
        print(f"[zenodo] put failed {key}: {r.status_code} {r.text[:200]}")
        return False
    r = requests.post(f"{API}/records/{record_id}/draft/files/{key}/commit",
                      headers=_headers(), timeout=120)
    if r.status_code >= 400:
        print(f"[zenodo] commit failed {key}: {r.status_code} {r.text[:200]}")
        return False
    print(f"[zenodo] uploaded {key} ({path.stat().st_size/1e6:.1f} MB)")
    return True


def publish(record_id):
    r = requests.post(f"{API}/records/{record_id}/draft/actions/publish",
                      headers=_headers(), timeout=300)
    if r.status_code >= 400:
        print(f"[zenodo] publish failed: {r.status_code} {r.text[:300]}")
        return False
    rec = r.json()
    print(f"[zenodo] published {record_id}: {rec.get('links', {}).get('doi')}")
    return True


def new_version(record_id):
    """Publish ``record_id`` (if needed) and open the next version draft."""
    try:
        publish(record_id)
    except Exception as exc:                     # noqa: BLE001
        print(f"[zenodo] publish before versioning failed: {exc}")
    r = requests.post(f"{API}/records/{record_id}/versions",
                      headers=_headers(), timeout=120)
    r.raise_for_status()
    rec = r.json()
    print(f"[zenodo] new version draft {rec['id']} "
          f"(concept {rec.get('conceptrecid') or rec.get('concept', '')})")
    return rec


def pending(root: Path, seen: dict):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        key = str(p.relative_to(root))
        size = p.stat().st_size
        if size == 0:            # Zenodo rejects empty files at commit
            continue
        sig = (size, int(p.stat().st_mtime))
        if seen.get(key) != sig:
            yield key, p, sig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", help="directory to mirror")
    ap.add_argument("--record-id", help="existing Zenodo draft/record id")
    ap.add_argument("--kind", choices=["analysis", "trajectories"],
                    default="analysis")
    ap.add_argument("--watch", type=float, default=0,
                    help="seconds between scans (0 = one pass)")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--bundle-states", action="store_true",
                    help="tar each <root>/<state>/ into _bundles/<state>.tar.gz")
    ap.add_argument("--single-file", help="upload one file (with --key)")
    ap.add_argument("--key", help="remote key for --single-file")
    ap.add_argument("--delete-after", action="store_true")
    ap.add_argument("--state", default=None)
    args = ap.parse_args(argv)

    if args.single_file:
        if not args.record_id or not args.key:
            raise SystemExit("--single-file needs --record-id and --key")
        src = Path(args.single_file)
        ok = upload(args.record_id, args.key, src)
        if ok and args.delete_after:
            src.unlink()
            print(f"[zenodo] deleted local {src}")
        return 0 if ok else 1

    if not args.root:
        raise SystemExit("--root is required unless --single-file is used")
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    state_path = Path(args.state) if args.state else root / ".zenodo_record.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    rec_id = args.record_id or state.get("record_id")
    if rec_id:
        print(f"[zenodo] using draft {rec_id}")
    else:
        rec = create_draft(args.kind)
        rec_id = rec["id"]
        state["record_id"] = rec_id
        state_path.write_text(json.dumps(state, indent=2))

    def refresh_bundles():
        """Bundle each analysed state, rebuilding when the state changed.

        Called every cycle: states finish while this watcher is alive, so a
        one-shot bundle pass would leave a stale empty archive for any state
        analysed after startup.
        """
        import tarfile
        bdir = root / "_bundles"
        bdir.mkdir(exist_ok=True)
        for st in sorted(p for p in root.iterdir() if p.is_dir()):
            if st.name.startswith("_"):
                continue
            # Only archive states whose post-processing is complete.
            if not (st / "analysis.json").exists():
                continue
            tgt = bdir / f"{st.name}.tar.gz"
            newest = max((f.stat().st_mtime for f in st.rglob("*")
                          if f.is_file()), default=0.0)
            if tgt.exists() and tgt.stat().st_mtime >= newest:
                continue
            print(f"[zenodo] bundling {st.name}", flush=True)
            tmp = Path(str(tgt) + ".tmp")
            with tarfile.open(tmp, "w:gz") as tf:
                tf.add(st, arcname=st.name)
            os.replace(tmp, tgt)

    if args.bundle_states:
        refresh_bundles()

    while True:
        if args.bundle_states:
            refresh_bundles()
        seen = state.setdefault("files", {})
        n = 0
        for key, path, sig in pending(root, seen):
            if key.startswith(".zenodo_record"):
                continue
            if key.startswith("_bundles/"):
                key = "states/" + key[len("_bundles/"):]
            if upload(rec_id, key, path):
                seen[key] = sig
                n += 1
                state_path.write_text(json.dumps(state, indent=2))
        print(f"[zenodo {time.strftime('%H:%M:%S')}] {n} file(s) uploaded")
        if args.publish:
            publish(rec_id)
            return 0
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
