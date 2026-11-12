#!/usr/bin/env bash
# Set up the Python side of the MACELES c1/c2 sampling pipeline.
#
#   scripts/setup_env.sh            # check the active interpreter
#   scripts/setup_env.sh --install  # pip install the requirements
#
# Environment:
#   PY        python interpreter (default: python3 on PATH)
#   PIP_ARGS  extra pip arguments (e.g. "--index-url ...")
set -euo pipefail

PY=${PY:-python3}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REQ="$HERE/../requirements.txt"

echo "== interpreter =="
command -v "$PY" || { echo "no $PY on PATH" >&2; exit 1; }
"$PY" - <<'PY'
import sys
print("python", sys.version.split()[0], "->", sys.executable)
PY

if [[ "${1:-}" == "--install" ]]; then
    echo "== installing $REQ =="
    "$PY" -m pip install ${PIP_ARGS:-} -r "$REQ"
fi

echo "== module check =="
"$PY" - <<'PY'
import importlib
need = ["numpy", "scipy", "ase", "matplotlib"]
missing = []
for m in need:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10s} {getattr(mod, '__version__', '?'):10s} ok")
    except Exception as exc:                       # noqa: BLE001
        print(f"  {m:10s} MISSING ({exc})")
        missing.append(m)
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing) +
                     "\nrun: scripts/setup_env.sh --install")
PY

echo "== optional speed-ups =="
"$PY" - <<'PY'
for m in ("pymatgen", "gsd", "ovito"):
    try:
        import importlib
        importlib.import_module(m)
        print(f"  {m:10s} available (optional)")
    except Exception:
        print(f"  {m:10s} not installed (optional)")
PY

echo "done."
