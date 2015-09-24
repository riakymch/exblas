#!/bin/bash

for i in $(seq 10 27); do
    ./../build/tests/test.exdot $i 1 > /dev/null
done
