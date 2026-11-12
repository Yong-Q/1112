#!/usr/bin/env bash
# Build the LAMMPS + Kokkos engine with the Symmetrix/MACELES-LES pair style.
#
#   scripts/build_engine.sh
#
# Environment (all optional):
#   SYMMETRIX_DIR   symmetrix-xl checkout          (default: $BZA/external-repos/symmetrix-xl)
#   LAMMPS_DIR      LAMMPS source tree             (default: $BZA/external-repos/lammps-stable_30Sep2026)
#   BUILD_DIR       build directory                (default: $SYMMETRIX_DIR/symmetrix/build-lammps-les)
#   CUDA_HOME       CUDA toolkit                   (default: /opt/matos-tools/cuda/12.9)
#   CUDA_ARCH       CMAKE_CUDA_ARCHITECTURES       (default: 80 = A100)
#   JOBS            parallel build jobs            (default: nproc)
#   CONDA_LIB       extra library dir (OpenBLAS)   (default: <toolchain>/conda/envs/matos-compute/lib)
#
# The MACELES-LES integration itself lives in
#   engine/polar-les/0001-maceles-complete-LES-integration-with-charge-respons.patch
# and is applied to a clean symmetrix-xl checkout before building.  Skip the
# patch step with APPLY_PATCH=0 if the checkout already contains it.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
BZA=${BZA:-$(cd -- "$REPO/.." && pwd -P)}

SYMMETRIX_DIR=${SYMMETRIX_DIR:-$BZA/external-repos/symmetrix-xl}
LAMMPS_DIR=${LAMMPS_DIR:-$BZA/external-repos/lammps-stable_30Sep2026}
BUILD_DIR=${BUILD_DIR:-$SYMMETRIX_DIR/symmetrix/build-lammps-les}
CUDA_HOME=${CUDA_HOME:-/opt/matos-tools/cuda/12.9}
CUDA_ARCH=${CUDA_ARCH:-80}
JOBS=${JOBS:-$(( $(nproc) < 16 ? $(nproc) : 16 ))}   # keep the login node usable
CONDA_LIB=${CONDA_LIB:-/opt/matos-tools/conda/envs/matos-compute/lib}
PATCH=${PATCH:-}
if [[ -z "$PATCH" ]]; then
for _d in "$REPO/../symmetrix-polar/engine/polar-les" \
          "$REPO/../symmetrix-polar/engine/maceles-les" \
          "$REPO/engine/polar-les" "$REPO/engine/maceles-les"; do
    if [[ -f "$_d/0001-maceles-complete-LES-integration-with-charge-respons.patch" ]]; then
        PATCH="$_d/0001-maceles-complete-LES-integration-with-charge-respons.patch"
        break
    fi
done
fi

echo "== 1. symmetrix checkout =="
if [[ ! -d "$SYMMETRIX_DIR" ]]; then
    echo "cloning symmetrix-xl into $SYMMETRIX_DIR"
    mkdir -p "$(dirname "$SYMMETRIX_DIR")"
    git clone git@github.com:bonan-group/symmetrix-xl.git "$SYMMETRIX_DIR"
fi
cd "$SYMMETRIX_DIR"
git rev-parse --short HEAD

echo "== 2. MACELES-LES patch =="
if [[ "${APPLY_PATCH:-1}" == "1" && -f "$PATCH" ]]; then
    if git apply --check "$PATCH" 2>/dev/null; then
        git apply "$PATCH"
        echo "applied $(basename "$PATCH")"
    elif git apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "patch already applied"
    else
        echo "WARNING: patch neither applies nor is applied; using the tree as-is" >&2
    fi
fi

echo "== 3. LAMMPS source =="
if [[ ! -d "$LAMMPS_DIR/src" ]]; then
    echo "ERROR: LAMMPS source not found at $LAMMPS_DIR" >&2
    echo "fetch LAMMPS stable_30Sep2026 (or 15Sep2022+) and set LAMMPS_DIR" >&2
    exit 1
fi
"$SYMMETRIX_DIR/pair_symmetrix/install.sh" "$LAMMPS_DIR"
rm -f "$LAMMPS_DIR"/src/*.bak

echo "== 4. cmake configure =="
export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CONDA_LIB:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_LIB:${LD_LIBRARY_PATH:-}"
mkdir -p "$BUILD_DIR"
cmake -S "$LAMMPS_DIR/cmake" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DBUILD_MPI=OFF \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=OFF \
    -DCMAKE_CXX_COMPILER="$LAMMPS_DIR/lib/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_INSTALL_PREFIX="$REPO/dist/lammps-cuda-sm${CUDA_ARCH}-nompi"

echo "== 5. build (-j $JOBS) =="
cmake --build "$BUILD_DIR" -j "$JOBS"

echo "== 6. result =="
ls -l "$BUILD_DIR/lmp"
echo "export LMP=$BUILD_DIR/lmp"
echo "note: link needs -lblas from $CONDA_LIB (LIBRARY_PATH above)."
