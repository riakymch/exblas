# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

cmake_minimum_required (VERSION 2.8.8)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckFunctionExists)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -fabi-version=0 -O3 -Wall")

# architecture specific flags
option (EXBLAS_GPU_AMD "By setting this option we enable compilation for AMD GPUs" OFF)
option (EXBLAS_GPU_NVIDIA "By setting this option we enable compilation for NVIDIA GPUs" OFF)
if (EXBLAS_GPU_AMD)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=att -DAMD")
    include_directories( "/opt/sdks/amd/current/include/" )
    link_directories ( "/opt/sdks/amd/current/lib/x86_64/" )
elseif (EXBLAS_GPU_NVIDIA)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel")
    include_directories( "/usr/local/cuda/include/" )
    link_directories ( "/usr/local/cuda/lib64/" )
endif (EXBLAS_GPU_AMD)

# enabling comparing with MPFR
option (EXBLAS_VS_MPFR "Use MPFR to compare the achived results for tests" OFF)
if (EXBLAS_VS_MPFR)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXBLAS_VS_MPFR")
    include_directories ("$ENV{GMP_HOME}/include/")
    include_directories ("$ENV{MPFR_HOME}/include/")
    link_directories ("$ENV{GMP_HOME}/lib64/")
    link_directories ("$ENV{MPFR_HOME}/lib64/")
    link_directories ("$ENV{GMP_HOME}/lib/")
    link_directories ("$ENV{MPFR_HOME}/lib/")
    set (EXTRA_LIBS ${EXTRA_LIBS} gmp)
    set (EXTRA_LIBS ${EXTRA_LIBS} mpfr)
endif (EXBLAS_VS_MPFR)

# add exblas
include_directories ("$ENV{EXBLAS_HOME}/include/")
link_directories ("$ENV{EXBLAS_HOME}/lib/")
set (EXTRA_LIBS ${EXTRA_LIBS} exblas)
set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)

# Testing
add_executable (test ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries (test ${EXTRA_LIBS})

