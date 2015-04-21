# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Recurse into the "cpu" and "gpu" subdirectories to build architecture specific library

# Compile for CPUs
option (EXBLAS_CPU "Create a CPU specific library" OFF)
# Compile for Intel Phi 
option (EXBLAS_MIC "Create a Intel Phi specific library" OFF)
# Compile for GPUs
option (EXBLAS_GPU "Create a GPU specific library" OFF)

if (EXBLAS_MIC)
   add_subdirectory (mic/blas1)
elseif (EXBLAS_GPU)
   add_subdirectory (gpu/blas1)
else (EXBLAS_CPU)
   add_subdirectory (cpu/blas1)
endif (EXBLAS_MIC)
