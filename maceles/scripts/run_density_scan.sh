#!/usr/bin/env bash
# End-to-end PRL-style density scan:
#   data -> shards -> MD (with per-state gzip) -> c1/c2 + structure -> figures
#
#   scripts/run_density_scan.sh                 # full local run
#   STAGE=data scripts/run_density_scan.sh      # only rebuild the input data
#   STAGE=post scripts/run_density_scan.sh      # only post-process
#
# Environment:
#   BZA          workspace root            (default: repo/..)
#   OUT          scan root                 (default: $BZA/symmetrix-polar/bench/maceles_density_scan)
#   PY           python                    (default: python3)
#   LMP          LAMMPS binary             (required for STAGE=run)
#   MODEL/CUBIN  MACELES checkpoint + cubin
#   SHARDS       number of shard manifests (default 4)
#   DUMP_EVERY   dump cadence in steps     (default 1000 = 0.5 ps at 0.5 fs)
#   EQUIL_STEPS  equilibration steps       (default 200000 = 100 ps)
#   STEPS        production steps          (default 2000000 = 1 ns)
#   STAGE        data|split|run|post|all   (default all)
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
BZA=${BZA:-$(cd -- "$REPO/.." && pwd -P)}
PY=${PY:-python3}
OUT=${OUT:-$BZA/symmetrix-polar/bench/maceles_density_scan}
SHARDS=${SHARDS:-4}
DUMP_EVERY=${DUMP_EVERY:-1000}
EQUIL_STEPS=${EQUIL_STEPS:-200000}
STEPS=${STEPS:-2000000}
STAGE=${STAGE:-all}
MODEL=${MODEL:-$BZA/symmetrix-polar/models/maceles_off_chno.json}
CUBIN=${CUBIN:-$BZA/symmetrix-polar/bench/q4_maceles_off/attempt-03/maceles_off.cubin}

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

if [[ "$STAGE" == "data" || "$STAGE" == "all" ]]; then
    echo "== data: 6 species x 10 densities -> $OUT =="
    "$PY" sampling/make_density_scan.py --outroot "$OUT" \
        --n-densities 10 --lo 0.2 --hi 2.2 \
        --steps "$STEPS" --dump-every "$DUMP_EVERY"
fi

if [[ "$STAGE" == "split" || "$STAGE" == "all" ]]; then
    echo "== split into $SHARDS shards =="
    "$PY" -m sampling.split_scan --root "$OUT" --shards "$SHARDS" \
        --attempt attempt-01 --dump-every "$DUMP_EVERY"
fi

if [[ "$STAGE" == "run" || "$STAGE" == "all" ]]; then
    : "${LMP:?set LMP to the built lammps binary (scripts/build_engine.sh)}"
    for i in $(seq 1 "$SHARDS"); do
        man="$OUT/shard_$i/attempt-01/manifest.csv"
        [[ -f "$man" ]] || { echo "missing $man" >&2; exit 1; }
        echo "== shard $i =="
        "$PY" sampling/les_batch.py \
            --manifest "$man" \
            --outroot "$OUT/shard_$i/attempt-01/samples" \
            --run --resume --no-analyze \
            --equil-steps "$EQUIL_STEPS" --steps "$STEPS" \
            --lmp "$LMP" --model "$MODEL" --cubin "$CUBIN" \
            --ld-library-path "${LD_LIBRARY_PATH:-}"
    done
fi

if [[ "$STAGE" == "post" || "$STAGE" == "all" ]]; then
    echo "== post-process: c1/c2 + structure + figures =="
    "$PY" -m sampling.scan_analysis --root "$OUT" --out "$OUT/analysis_scan"
fi

echo "done: $OUT/analysis_scan"
