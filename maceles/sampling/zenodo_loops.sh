#!/usr/bin/env bash
# Supervisor for the three post-processing / Zenodo loops.
#
#   scripts/zenodo_loops.sh            # run in the foreground (or under a manager)
#
# Anything that dies is restarted after 60 s; all three loops are idempotent
# (state files record what was uploaded), so a restart never re-sends data and
# never loses a finished trajectory (a file is deleted only after Zenodo
# returned 2xx).
#
# Environment:
#   SCAN_ROOT   default <bza>/symmetrix-polar/bench/maceles_density_scan
#   ANA_RECORD  Zenodo record for analysis products (default 23901520)
#   TRAJ_RECORD Zenodo record for raw trajectories  (default 23901522)
#   ZENODO_TOKEN  read from .secrets/zenodo_token when unset
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
BZA=${BZA:-$(cd -- "$REPO/.." && pwd -P)}
PY=${PY:-/opt/matos-tools/conda/envs/matos-compute/bin/python}
SCAN_ROOT=${SCAN_ROOT:-$BZA/symmetrix-polar/bench/maceles_density_scan}
ANA=${SCAN_ROOT}/analysis_scan
ANA_RECORD=${ANA_RECORD:-23901520}
TRAJ_RECORD=${TRAJ_RECORD:-23901522}
LOGS=${LOGS:-$ANA}

if [[ -z "${ZENODO_TOKEN:-}" && -f "$BZA/.secrets/zenodo_token" ]]; then
    ZENODO_TOKEN=$(cat "$BZA/.secrets/zenodo_token")
fi
export ZENODO_TOKEN PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO" || exit 1          # so "python -m sampling" resolves to this repo
mkdir -p "$LOGS"

supervise() {
    local name=$1; shift
    while true; do
        echo "[sup $(date -u +%H:%M:%S)] start $name" >> "$LOGS/supervisor.log"
        "$@" >> "$LOGS/$name.log" 2>&1
        echo "[sup $(date -u +%H:%M:%S)] $name exited rc=$?; restart in 60 s" \
            >> "$LOGS/supervisor.log"
        sleep 60
    done
}

supervise watch "$PY" -m sampling.scan_watch \
    --root "$SCAN_ROOT" --out "$ANA" --interval 900 --keep-traj &
supervise mirror "$PY" -m sampling.zenodo_upload \
    --root "$ANA" --kind analysis --record-id "$ANA_RECORD" \
    --bundle-states --watch 1800 &
supervise traj "$PY" -m sampling.zenodo_batch_traj \
    --root "$SCAN_ROOT" --record-id "$TRAJ_RECORD" --batch 3 --watch 1800 \
    --archive-dir "$SCAN_ROOT/traj_archive" &
supervise sync "$HERE/sync_postproc.sh" 1800 &
wait
