#!/bin/sh

nbMax=32768
step=2

for i in $(seq 5 14);
do
    ./../build/tests/test.extrsv U $i 1 > /dev/null 
done
