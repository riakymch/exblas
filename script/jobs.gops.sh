#!/bin/bash
#
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8
#PBS -l place=excl:group=socket
#PBS -l walltime=01:00:00
#PBS -N exsum
#PBS -j oe
#PBS -o exsum.gops.mesu.15.07.28.1.txt

# load modules
. /usr/share/modules/init/sh

module unload intel-mpi
module unload intel-cmkl-15/15.0.0.090

module load gcc/4.9.0
module load intel-tbb/13.0.1.117
module load intel-compilers-15/15.0.0.090
module load mpt

# go to directory in which the job_script_file reside
cd $PBS_O_WORKDIR
cd build/tests

export OMP_NUM_THREADS=8
export KMP_AFFINITY=disabled
#export MPI_OPENMP_INTEROP=1

module list

for i in {10..29}; do
    omplace ./test.exsum $i 8 > /dev/null
    #./test.exsum $i 8 > /dev/null
done

