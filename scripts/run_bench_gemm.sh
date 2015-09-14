#!/bin/sh

nbMax=32768
step=2

for i in $(seq 5 11);
do
    ./../build/tests/test.exgemm $i $i $i 1 > /dev/null 
done
