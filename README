ExBLAS -- Exact (fast, accurate, and reproducible) BLAS library v1.0
=============================================
ExBLAS aims at providing new algorithms and implementations for fundamental 
linear algebra operations -- like those included in the BLAS library -- that 
deliver reproducible and accurate results with small or without losses to their 
performance on modern parallel architectures such as Intel desktop and server 
processors, Intel Xeon Phi co-processors, and GPU accelerators. We construct 
our approach in such a way that it is independent from data partitioning, 
order of computations, thread scheduling, or reduction tree schemes.

The current version of the ExBLAS library is equipped with 
  * ExSUM -- Reproducible and accurate parallel reduction (parallel summation);
  * ExDOT -- Reproducible and accurate parallel dot product;
  * ExGEMV -- Reproducible and accurate parallel matrix-vector product for various sizes (m = n, m >= n, and m <= n) for both transpose and non-transpose matrices;
  * ExTRSV -- Reproducible and accurate parallel triangular solver for both transpose and non-transpose, lower and unit triangular matrices with unit and non-unit diagonals;
  * ExGEMM -- Reproducible and accurate parallel matrix-matrix multiplication for squeare matrices for a moment.
These routines can be executed on a set of architectures: Intel Core i7 and 
Sandy Bridge processors; Intel Xeon Phi co-processors; both AMD and NVIDIA 
GPUs using the OpenCL framework.


Requirements
=============================================
In order to use ExBLAS, the following software is needed:
  * [Required] Cmake of version 2.8.8 or higher
  * [Required] For CPUs, support of AVX instructions
  * [Required] For CPUs, Intel TBB library of version 4.0 or higher
  * [Required] For CPUs, support of C++11
  * [Required] For MIC, Intel C/C++ compilers
  * [Required] For GPUs only, OpenCL of version 1.1 or higher
  * [Optional] GMP of version 6.0
  * [Optional] MPFR of version 3.1.2
The later two are required to verify the accuracy and reproducibility of the 
results.


Installation
=============================================
Preliminaries
---------------------------------------------
* Download the latest version of source code from
  exblas.lip6.fr
* Extract the sources from a tar.gz archive (tar -zxvf) to {EXBLAS_ROOT} folder
* Create a build dir in the root of the library and move there

Compilation Options
---------------------------------------------
* -DEXBLAS_MPI=ON -- enables compilation with MPI. By default, only shared-memory 
   parallelism is activated
* -DEXBLAS_MIC=ON -- enables compilation for Intel MIC architectures
* -DEXBLAS_GPU=ON -- enables compilation for GPUs
   -DEXBLAS_GPU_AMD=ON -- for AMD GPUs
   -DEXBLAS_GPU_NVIDIA=ON -- for NVIDIA GPUs
* -DEXBLAS_VS_MPFR=ON -- compares the results against the ones produced by MPFR

Compilation
---------------------------------------------
* For Intel CPU with AVX instructions: CC=icc CXX=icpp cmake ..
                                       make
                                       make install

* With MPI: CC=mpicc CXX=mpicxx cmake -DEXBLAS_MPI=ON ..
            make
            make install

* For GPU: CC=mpicc CXX=mpicxx cmake -DEXBLAS_GPU=ON -DEXBLAS_AMD|NVIDIA_GPU=ON ..
           make
           make install

* For MIC: CC=mpicc CXX=mpicxx cmake -DEXBLAS_MIC=ON ..
           make
           make install

Testing
---------------------------------------------
* make test -- it will run several tests using naive, loguniform, and ill-cond
               number distribution comparing the results either against the ones
               produced by superaccumulators only or by MPFR

* All tests are stored in the tests dir

Documentation
---------------------------------------------
* [Required] Doxygen of version 1.8.0

* [Optional] LaTeX in case you want to generate documentation in pdf format

* -DEXBLAS_DOC=ON -- enables generationg of the ExBLAS documentation using Doxygen and LaTeX

* make doc -- command to generate an html version of the ExBLAS documentation

* make pdf -- command to generate a pdf version of the ExBLAS documentation


Examples
=============================================
In the example dir you can find a simple test that shows how the ExBLAS routines 
can be called and how the ExBLAS library can be linked from your project.

