#!/bin/bash
#
#PBS -l select=1:ncpus=8:mpiprocs=1:ompthreads=8
#PBS -l place=excl:group=socket
#PBS -l walltime=01:00:00
#PBS -N reprosum_np1
#PBS -j oe
#PBS -o ../results/mpi.np1.n24.barrier.demmel.15.07.24.txt

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
cd ../src

export OMP_NUM_THREADS=8
export KMP_AFFINITY=disabled
export MPI_OPENMP_INTEROP=1

module list

for i in {10..29}; do
    mpirun -np 1 omplace ./longacc.new $i 8 > /dev/null
done

#for j in {0..10}; do
#    KMP_AFFINITY=disabled MPI_OPENMP_INTEROP=1 mpirun -np 1 omplace ./reprosum 21 1 1 0 y
#done
