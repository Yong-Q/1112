#!/usr/bin/env bash
# End-to-end feasibility run on a 96-atom water frame.
# Edit the two model paths and the structure path before running.
set -euo pipefail
STRUCTURE="${STRUCTURE:-water96.xyz}"
OFF_MODEL="${OFF_MODEL:?set OFF_MODEL to MACE-OFF23_medium.model}"
POLAR_MODEL="${POLAR_MODEL:?set POLAR_MODEL to MACE-POLAR-1-M.model}"
OUT="${OUT:-runs/water96}"

off-polar-run \
  --structure "$STRUCTURE" \
  --model "$OFF_MODEL" \
  --out "$OUT/off_traj.xyz" \
  --temperature 300 --ps 5 --equil-ps 0.5 --frame-ps 0.1

off-polar-reweight \
  --traj "$OUT/off_traj.xyz" \
  --source-model "$OFF_MODEL" \
  --polar-model "$POLAR_MODEL" \
  --temperature 300 --outdir "$OUT/reweight"
