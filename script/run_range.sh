#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$HOME/lib"
export KMP_AFFINITY=compact
export OMP_NUM_THREADS=236

#N=23
N=21

micnativeloadex ./../build/tests/test.exsum -a "$N 1" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 10" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 20" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 30" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 40" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 50" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 60" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 70" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 80" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 90" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 100" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 120" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 140" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 160" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 180" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 200" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 250" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 300" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 400" > /dev/null
micnativeloadex ./../build/tests/test.exsum -a "$N 500" > /dev/null

