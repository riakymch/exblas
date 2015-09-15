#!/bin/bash

nbMax=2048
step=64

for ((i=64; i<=${nbMax}; i+=${step}))
do
    ./../build/tests/test.exgemm $i $i $i 1 > /dev/null 
done
