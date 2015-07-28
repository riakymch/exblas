#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/lib"
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=240
export MIC_OMP_NUM_THREADS=240

for i in $(seq 16 29); do
    micnativeloadex ./../build/tests/test.exsum -a "$i 1" > /dev/null
done
