#!/usr/bin/env bash
# One-shot setup for the PyTorch MACELES reference.
#
#   scripts/setup_maceles_pt.sh
#   python scripts/maceles_reference.py --model assets/maceles_off/MACELES-OFF_small.model --eval frame.xyz
#
# Clones the two public sources that carry the MACELES class and the LES module
# at the exact commits this workspace used, installs them, and verifies that the
# checkpoint loads.
#
#   https://github.com/ChengUCB/mace.git   main  (mace.modules.extensions.MACELES)
#   https://github.com/ChengUCB/les.git    main  (les.Les)
# The toolchain checkout in this container is the same public repo at a279281;
# set MACE_REF=a2792819... if you want that exact revision.
#
# Environment:
#   PY         python interpreter (default: python3)
#   DEST       checkout root (default: <repo>/build/maceles-pt)
#   MODEL      checkpoint to verify (default assets/maceles_off/MACELES-OFF_small.model)
#   NO_INSTALL 1 = only clone, do not pip install
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
PY=${PY:-python3}
DEST=${DEST:-$REPO/build/maceles-pt}
MACE_REPO=${MACE_REPO:-https://github.com/ChengUCB/mace.git}
MACE_REF=${MACE_REF:-main}          # pin a SHA if you need the exact toolchain revision
LES_REPO=${LES_REPO:-https://github.com/ChengUCB/les.git}
LES_REF=${LES_REF:-main}
MODEL=${MODEL:-$REPO/assets/maceles_off/MACELES-OFF_small.model}

clone_pin() {
    local repo=$1 ref=$2 dir=$3
    if [[ ! -d "$dir/.git" ]]; then
        echo "== clone $repo -> $dir"
        git clone --filter=blob:none "$repo" "$dir"
    fi
    echo "== checkout $dir @ $ref"
    if ! git -C "$dir" fetch --depth 1 origin "$ref"; then
        echo "WARNING: cannot fetch '$ref'; falling back to the default branch" >&2
        git -C "$dir" fetch --depth 1 origin HEAD
    fi
    git -C "$dir" checkout --detach FETCH_HEAD
}

clone_pin "$MACE_REPO" "$MACE_REF" "$DEST/mace-fork"
clone_pin "$LES_REPO" "$LES_REF" "$DEST/les-fork"

# The fork vendors ~153 MB of foundation-model weights that the MACELES path
# does not need.  Drop them from the working tree (git keeps them recoverable).
FOUND="$DEST/mace-fork/mace/calculators/foundations_models"
if [[ -d "$FOUND" && "${KEEP_FOUNDATION_MODELS:-0}" != "1" ]]; then
    echo "== removing $FOUND (set KEEP_FOUNDATION_MODELS=1 to keep)"
    rm -rf "$FOUND"
fi

if [[ "${NO_INSTALL:-0}" != "1" ]]; then
    echo "== pip install (editable) =="
    "$PY" -m pip install -e "$DEST/mace-fork"
    "$PY" -m pip install -e "$DEST/les-fork"
fi

export PYTHONPATH="$DEST/mace-fork:$DEST/les-fork/src:${PYTHONPATH:-}"
echo "== verify =="
"$PY" "$REPO/scripts/maceles_reference.py" --model "$MODEL" --check

cat <<EOF

setup done.
  export PYTHONPATH="$DEST/mace-fork:$DEST/les-fork/src:\$PYTHONPATH"
  $PY scripts/maceles_reference.py --model "$MODEL" --eval frame.xyz --device cuda
EOF
