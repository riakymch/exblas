#!/bin/bash
#
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8
#PBS -l place=excl:group=socket
#PBS -l walltime=01:00:00
#PBS -N exsum
#PBS -j oe
#PBS -o exsum.range.mesu.15.09.16.txt

# load modules
. /usr/share/modules/init/sh

module unload mpt
module unload intel-mpi
module unload intel-cmkl-15/15.0.0.090
module unload intel-compilers-15/15.0.0.090

module load gcc/4.9.0
module load intel-tbb/13.0.1.117

# go to directory in which the job_script_file reside
cd $PBS_O_WORKDIR
cd ../build/tests

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

module list

N=21

./test.exsum $N 1 > /dev/null
./test.exsum $N 10 > /dev/null
./test.exsum $N  20 > /dev/null
./test.exsum $N  30 > /dev/null
./test.exsum $N  40 > /dev/null
./test.exsum $N  50 > /dev/null
./test.exsum $N  60 > /dev/null
./test.exsum $N  70 > /dev/null
./test.exsum $N  80 > /dev/null
./test.exsum $N  90 > /dev/null
./test.exsum $N  100 > /dev/null
./test.exsum $N  120 > /dev/null
./test.exsum $N  140 > /dev/null
./test.exsum $N  160 > /dev/null
./test.exsum $N  180 > /dev/null
./test.exsum $N  200 > /dev/null
./test.exsum $N  250 > /dev/null
./test.exsum $N  300 > /dev/null
./test.exsum $N  400 > /dev/null
./test.exsum $N  500 > /dev/null
