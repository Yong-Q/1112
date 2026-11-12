#!/usr/bin/env bash
# Assemble a self-contained "unpack and run" MACELES/LAMMPS engine bundle.
#
#   scripts/make_lmp_bundle.sh                # -> dist/maceles-lmp-env-<ver>.tar.gz
#   scripts/make_lmp_bundle.sh --check FILE   # verify an existing tarball
#
# Environment:
#   LMP         LAMMPS binary to ship   (default: $REPO/bin/lmp, else $BUILD_DIR/lmp)
#   BUILD_DIR   build tree              (default: ../build-lammps-les)
#   VERSION     bundle version suffix   (default: cu<cuda>-sm<arch>-<date>)
#   OUTDIR      output directory        (default: $REPO/dist)
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO=$(cd -- "$HERE/.." && pwd -P)
OUTDIR=${OUTDIR:-$REPO/dist}
TPL=$REPO/packaging/lmp-env

if [[ "${1:-}" == "--check" ]]; then
    tar=${2:?usage: $0 --check dist/maceles-lmp-env-*.tar.gz}
    tmp=$(mktemp -d)
    tar xzf "$tar" -C "$tmp"
    ( cd "$tmp"/maceles-lmp-env-* && sha256sum -c MANIFEST.sha256 )
    rm -rf "$tmp"
    echo "bundle OK: $tar"
    exit 0
fi

LMP=${LMP:-}
if [[ -z "$LMP" ]]; then
    for c in "$REPO/bin/lmp" "$REPO/../build-lammps-les/lmp"; do
        [[ -x "$c" ]] && LMP=$c && break
    done
fi
[[ -x "${LMP:-}" ]] || { echo "no lmp binary; set LMP=/path/to/lmp" >&2; exit 1; }

cuda_ver=$(readlink -f "$(dirname "$LMP")/../lib64/libcudart.so.12" 2>/dev/null \
    || readlink -f /usr/local/cuda/lib64/libcudart.so.12 2>/dev/null || echo "")
cuda_ver=$(basename "${cuda_ver:-libcudart.so.12.unknown}" | sed 's/^libcudart\.so\.//')
arch=$( { strings "$LMP" | grep -oE 'sm_(80|90)' | head -1; } || true )
arch=${arch:-sm_80}
VERSION=${VERSION:-"cu${cuda_ver}-${arch}-$(date +%Y%m%d)"}
NAME=maceles-lmp-env-$VERSION
STAGE=$OUTDIR/$NAME

echo "== assemble $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{bin,lib,assets/maceles_off,sampling,scripts,jobs,examples/smoke}

echo "== binary"
install -m 755 "$LMP" "$STAGE/bin/lmp"

echo "== runtime libraries"
# Everything except glibc itself and the NVIDIA driver stub.
# cublas/cublasLt are ~0.85 GB and every CUDA-12 node already has them;
# env.sh locates the toolkit at run time instead.
skip='^(linux-vdso|ld-linux|libc\.so|libm\.so|libmvec|libpthread|librt|libdl|libcuda\.so|libnvidia|libcublas)'
copied=()
while read -r soname path; do
    [[ -z "$soname" || "$soname" =~ $skip ]] && continue
    real=$(readlink -f "$path")
    [[ -f "$real" ]] || continue
    cp -f "$real" "$STAGE/lib/$soname"
    copied+=("$soname")
done < <(ldd "$LMP" | awk '/=> \//{print $1, $3}')
for extra in libstdc++.so.6 libgcc_s.so.1; do
    [[ -e "$STAGE/lib/$extra" ]] && continue
    real=$(readlink -f "/lib/x86_64-linux-gnu/$extra" 2>/dev/null) || continue
    [[ -f "$real" ]] && cp -f "$real" "$STAGE/lib/$extra" && copied+=("$extra")
done
printf '   %s\n' "${copied[@]}"

echo "== templates, assets, pipeline"
cp -a "$TPL"/env.sh "$TPL"/README.md "$TPL"/bin "$TPL"/jobs "$TPL"/examples "$STAGE"/
cp -a "$REPO"/assets/maceles_off/maceles_off_chno.json \
      "$REPO"/assets/maceles_off/maceles_off.cubin "$STAGE/assets/maceles_off/"
cp -a "$REPO"/sampling/*.py "$REPO"/sampling/README.md "$STAGE/sampling/"
cp -a "$REPO"/scripts/run_density_scan.sh "$STAGE/scripts/"
cp -a "$REPO"/requirements.txt "$STAGE/" 2>/dev/null || true
chmod +x "$STAGE"/env.sh "$STAGE"/bin/lmp-run "$STAGE"/jobs/*.sh "$STAGE"/jobs/slurm.sbatch \
         "$STAGE"/examples/smoke/run.sh "$STAGE"/scripts/run_density_scan.sh

echo "== build info"
{
    echo "bundle        $NAME"
    echo "built         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "lmp source    $(strings "$LMP" | grep -m1 -e 'Large-scale Atomic' || true)"
    echo "lmp sha256    $(sha256sum "$LMP" | cut -d' ' -f1)"
    echo "cuda runtime  $cuda_ver"
    echo "cuda arch     $arch"
    echo "host gcc      $(${CC:-gcc} -dumpversion 2>/dev/null || echo unknown)"
    echo "host kernel   $(uname -srm)"
    echo "model sha256  $(sha256sum "$STAGE/assets/maceles_off/maceles_off_chno.json" | cut -d' ' -f1)"
    echo "cubin sha256  $(sha256sum "$STAGE/assets/maceles_off/maceles_off.cubin" | cut -d' ' -f1)"
} > "$STAGE/BUILD_INFO.txt"
cat "$STAGE/BUILD_INFO.txt"

echo "== manifest"
( cd "$STAGE" && find . -type f ! -name MANIFEST.sha256 -printf '%P\n' | sort \
    | xargs sha256sum > MANIFEST.sha256 )

echo "== tarball"
mkdir -p "$OUTDIR"
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -czf "$OUTDIR/$NAME.tar.gz" -C "$OUTDIR" "$NAME"
( cd "$OUTDIR" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256" )
ls -lh "$OUTDIR/$NAME.tar.gz" "$OUTDIR/$NAME.tar.gz.sha256"
echo "export BUNDLE=$STAGE"
