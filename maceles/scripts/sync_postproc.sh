#!/usr/bin/env bash
# Keep github.com/Yong-Q/new in sync with the post-processing code in this repo.
#
#   scripts/sync_postproc.sh [interval_seconds]
#
# Every pass: copy the post-processing modules (and their helpers) from
# test/sampling into a clone of Yong-Q/new, commit if anything changed, push.
set -uo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
SRC=${SRC:-$REPO/sampling}
WORK=${WORK:-/workspace/group/shared/bza/postproc_repo}
INTERVAL=${1:-1800}
FILES="scan_analysis pair_correlations intramolecular scan_watch zenodo_upload zenodo_batch_traj cdft_dataset analyze_traj_orientation trajectory direct_correlation les_c1c2 les_batch __init__"

export GIT_SSH_COMMAND="ssh -i /workspace/group/shared/.ssh/matos_github_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/workspace/group/shared/.ssh/known_hosts -o StrictHostKeyChecking=yes -o BatchMode=yes"

if [[ ! -d "$WORK/.git" ]]; then
    git clone git@github.com:Yong-Q/new.git "$WORK" || exit 1
fi

while true; do
    for f in $FILES; do cp "$SRC/$f.py" "$WORK/sampling/" 2>/dev/null; done
    cp "$REPO/scripts/zenodo_loops.sh" "$WORK/scripts/" 2>/dev/null
    cp "$REPO/requirements.txt" "$WORK/" 2>/dev/null
    cd "$WORK" || exit 1
    if [[ -n "$(git status --porcelain)" ]]; then
        git add -A
        git -c user.name="Yong-Q" -c user.email="yong-q@users.noreply.github.com" \
            commit -q -m "sync post-processing $(date -u +%Y-%m-%dT%H:%MZ)"
        git push -q origin HEAD:main && echo "[sync $(date -u +%H:%M:%S)] pushed"
    else
        echo "[sync $(date -u +%H:%M:%S)] no changes"
    fi
    sleep "$INTERVAL"
done
