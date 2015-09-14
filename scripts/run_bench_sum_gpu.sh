#!/bin/bash

for i in $(seq 16 29); do
    ./../build/tests/test.exsum $i 1 > /dev/null
done
