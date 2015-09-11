#!/bin/bash
#
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8
#PBS -l place=excl:group=socket
#PBS -l walltime=00:40:00
#PBS -N exsum
#PBS -j oe
#PBS -o exsum.gops.mesu.15.09.11.txt

# load modules
. /usr/share/modules/init/sh

module unload intel-mpi
module unload intel-cmkl-15/15.0.0.090
module unload intel-compilers-15/15.0.0.090
module unload mpt

module load gcc/4.9.0
module load intel-tbb/13.0.1.117

# go to directory in which the job_script_file reside
cd $PBS_O_WORKDIR
cd ../build/tests

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

module list

for i in {10..29}; do
    ./test.exsum $i 1 > /dev/null
done

