#!/bin/bash

file=../results/NbElements_VS_GBs_NVIDIA_2015_09_14.dat

touch $file
echo -n "" > $file

for i in $(seq 16 29); do
    ./../build/tests/test.exsum $i 1 > /dev/null
done
