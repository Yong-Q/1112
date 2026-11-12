#!/usr/bin/env bash
# Installation smoke test: 64-water box, 500 MD steps, c1/c2 + structure.
#
#   LMP=/path/to/lmp MODEL=/path/to/maceles_off_chno.json CUBIN=/path/to/cubin \
#       scripts/smoke_test.sh
#
# Needs a GPU.  Takes ~2 minutes.  Everything lands in ./smoke.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
PY=${PY:-python3}
: "${LMP:?set LMP to the LAMMPS binary built by scripts/build_engine.sh}"
: "${MODEL:?set MODEL to the MACELES checkpoint (.json)}"
: "${CUBIN:?set CUBIN to the matching .cubin}"
OUT=${OUT:-$REPO/smoke}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "== 1. build a 64-molecule water box =="
"$PY" -m sampling.make_smoke_box --out "$OUT"

echo "== 2. MD (500 steps, dump every 100) =="
"$PY" sampling/les_batch.py \
    --manifest "$OUT/manifest.csv" \
    --outroot "$OUT/samples" \
    --run --resume --no-analyze \
    --equil-steps 200 --steps 500 \
    --lmp "$LMP" --model "$MODEL" --cubin "$CUBIN" \
    --ld-library-path "${LD_LIBRARY_PATH:-}"

echo "== 3. post-process =="
"$PY" -m sampling.scan_analysis --root "$OUT" --out "$OUT/analysis_scan" \
    --frames 10 --c2-stride 1 --c2-bins 40

echo "== 4. result =="
cat "$OUT/analysis_scan/scan_summary.csv"
ls "$OUT/analysis_scan/figures"
echo "smoke test OK"
