#!/bin/bash
set -e
source /etc/profile.d/modules.sh
module purge
module use /publicfs10/fs10-share/soft/share-soft/modulefiles
module load gcc/14.2.0
module load openmpi/4.1.8-gcc14.2.0
export LD_LIBRARY_PATH=/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:$LD_LIBRARY_PATH
echo "NODE=$(hostname) CPU=$(nproc) NP=$NP"
cd /publicfs10/fs10-m9/home/m9s005258/qy/cdft/new
PY=/publicfs10/fs10-m9/home/m9s005258/qy/venv-mace/bin/python
$PY sampling/les_batch.py \
  --manifest /publicfs10/fs10-m9/home/m9s005258/qy/cdft/_smoke/manifest.csv \
  --outroot /publicfs10/fs10-m9/home/m9s005258/qy/cdft/_smoke/samples \
  --run --resume --no-analyze \
  --equil-steps 50 --steps 300 --dump-every 100 \
  --gpus 0 --np-ranks ${NP:-2} --omp-threads 1 \
  --host-artifact /publicfs10/fs10-m9/home/m9s005258/.cache/symmetrix/execution-jit/artifacts/8d00f0566cc8a5f6a603befb0eab51f47d3a025c19c9b9d92e0731eb351e84f7/factorized_host_plugin_gen11_src0c65a7c611c15feff220748dc034a7a963176ec834be421668b0f9c04a5ca982.so \
  --lmp /publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp \
  --model /publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/maceles_off_chno.json \
  --cubin /publicfs10/fs10-m9/home/m9s005258/qy/test/assets/maceles_off/maceles_off.cubin \
  --ld-library-path "/publicfs10/fs10-m9/home/m9s005258/qy/deps/blas/lib:/publicfs10/fs10-share/soft/share-soft/openmpi/openmpi-4.1.8/install-new/lib:/publicfs10/fs10-share/soft/share-soft/gcc/gcc-14.2.0/install/lib64"
