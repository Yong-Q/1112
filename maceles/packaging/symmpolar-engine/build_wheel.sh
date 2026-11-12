#!/usr/bin/env bash
# Build the `symmpolar-engine` CUDA wheel from a symmetrix-xl checkout.
#
#   SYMMPOLAR_CUDA_ARCH=80 scripts/../packaging/symmpolar-engine/build_wheel.sh [SYMMETRIX_DIR]
#
# Compiles Kokkos-CUDA + the Symmetrix python extension; needs nvcc (CUDA
# 12.x) but no GPU.  ~15-30 min.  The wheel is written to ./dist.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/../.." && pwd -P)
SYMMETRIX_REPO=${SYMMETRIX_REPO:-https://github.com/Yong-Q/symmetrix-xl_polar.git}
# Default: the MACELES/LES fork (Yong-Q/symmetrix-xl_polar, branch
# `polar-les` = upstream 6c5f73b + one LES commit).  Set SYMMETRIX_REPO to
# the upstream URL and SYMMETRIX_REF=6c5f73b to build upstream + patch instead.
SYMMETRIX_REF=${SYMMETRIX_REF:-polar-les}
SYMMETRIX_DIR=${1:-${SYMMETRIX_DIR:-}}
if [[ -z "$SYMMETRIX_DIR" ]]; then
    SYMMETRIX_DIR=$REPO/build/symmetrix-xl
fi
CUDA_ARCH=${SYMMPOLAR_CUDA_ARCH:-80}
PY=${PY:-python3}
OUT=${OUT:-$REPO/dist}

# self-contained: fetch exactly the base the patch applies to
if [[ ! -f "$SYMMETRIX_DIR/symmetrix/pyproject.toml" ]]; then
    echo "fetching $SYMMETRIX_REPO @ $SYMMETRIX_REF -> $SYMMETRIX_DIR"
    mkdir -p "$(dirname "$SYMMETRIX_DIR")"
    git clone --filter=blob:none --no-checkout "$SYMMETRIX_REPO" "$SYMMETRIX_DIR"
    git -C "$SYMMETRIX_DIR" fetch --depth 1 origin "$SYMMETRIX_REF"
    git -C "$SYMMETRIX_DIR" checkout --detach FETCH_HEAD
    git -C "$SYMMETRIX_DIR" submodule update --init --depth 1 --recursive
fi

PATCH=""
for _d in "$REPO/../symmetrix-polar/engine/polar-les" \
          "$REPO/../symmetrix-polar/engine/maceles-les" \
          "$REPO/engine/polar-les" "$REPO/engine/maceles-les"; do
    if [[ -f "$_d/0001-maceles-complete-LES-integration-with-charge-respons.patch" ]]; then
        PATCH="$_d/0001-maceles-complete-LES-integration-with-charge-respons.patch"
        break
    fi
done
if [[ ! -d "$SYMMETRIX_DIR/.git" ]]; then
    echo "no .git in $SYMMETRIX_DIR (source snapshot); assuming LES is already in the tree"
elif [[ -f "$PATCH" ]]; then
    cd "$SYMMETRIX_DIR"
    if git apply --check "$PATCH" >/dev/null 2>&1; then
        git apply "$PATCH"; echo "applied LES patch"
    elif git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
        echo "LES patch already applied"
    else
        echo "WARNING: LES patch neither applies nor is applied" >&2
    fi
fi

# The validated engine is upstream 6c5f73b + the LES commit (fork branch
# `polar-les`).  Building the fork's `main` would silently drop LES.
if [[ ! -f "$SYMMETRIX_DIR/libsymmetrix/source/les_ewald.hpp" ]]; then
    echo "ERROR: $SYMMETRIX_DIR has no LES (libsymmetrix/source/les_ewald.hpp" >&2
    echo "       missing).  Check out the polar-les branch, not main:" >&2
    echo "       git -C $SYMMETRIX_DIR checkout polar-les" >&2
    exit 1
fi
if [[ -d "$SYMMETRIX_DIR/.git" ]]; then
    echo "engine tree: $(git -C "$SYMMETRIX_DIR" log --oneline -1)"
else
    echo "engine tree: source snapshot (no .git)"
fi

BACKUP=$(mktemp)
cp "$SYMMETRIX_DIR/symmetrix/pyproject.toml" "$BACKUP"
trap 'cp "$BACKUP" "$SYMMETRIX_DIR/symmetrix/pyproject.toml"; rm -f "$BACKUP"' EXIT

cp "$HERE/pyproject.toml" "$SYMMETRIX_DIR/symmetrix/pyproject.toml"
if [[ ! -f "$SYMMETRIX_DIR/libsymmetrix/external/kokkos/CMakeLists.txt" ]]; then
    echo "Kokkos submodule missing; run: git -C $SYMMETRIX_DIR submodule update --init --depth 1" >&2
    exit 1
fi
NVCC_WRAPPER="$SYMMETRIX_DIR/libsymmetrix/external/kokkos/bin/nvcc_wrapper"
if [[ ! -x "$NVCC_WRAPPER" ]]; then
    echo "nvcc_wrapper not found at $NVCC_WRAPPER" >&2
    exit 1
fi
export PATH="$(dirname "${CUDACXX:-/usr/local/cuda/bin/nvcc}"):$PATH"
mkdir -p "$OUT"
echo "== building symmpolar-engine for sm_${CUDA_ARCH} =="
SYMMPOLAR_CUDA_ARCH="$CUDA_ARCH" \
SYMMPOLAR_NVCC_WRAPPER="$NVCC_WRAPPER" \
CUDACXX="${CUDACXX:-$(command -v nvcc || true)}" \
    "$PY" -m pip wheel "$SYMMETRIX_DIR/symmetrix" \
        --no-deps --no-build-isolation -w "$OUT"

ls -lh "$OUT"/symmpolar_engine-*.whl
echo "install: pip install $OUT/symmpolar_engine-*.whl"
