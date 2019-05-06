# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Generate GPU specific library relying upon OpenCL

# add the EXBLAS library?
option (USE_EXBLAS "Use ExBLAS provided reproducible and accurate implementations of the BLAS routines" ON)
if (USE_EXBLAS)
  include_directories ("${PROJECT_SOURCE_DIR}/include")
  include_directories ("${PROJECT_SOURCE_DIR}/src/common")
  include_directories ("${PROJECT_SOURCE_DIR}/src/gpu")
  include_directories ("${PROJECT_BINARY_DIR}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} exblas)
endif (USE_EXBLAS)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -fabi-version=0 -O3 -Wall -Wno-comment -Wno-maybe-uninitialized")

# architecture specific flags
option (EXBLAS_GPU_AMD "By setting this option we enable compilation for AMD GPUs" OFF)
option (EXBLAS_GPU_NVIDIA "By setting this option we enable compilation for NVIDIA GPUs" OFF)
if (EXBLAS_GPU_AMD)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=att -DAMD")
    include_directories( "/opt/sdks/amd/current/include/" )
    link_directories ( "/opt/sdks/amd/current/lib/x86_64/" )
elseif (EXBLAS_GPU_NVIDIA)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel")
    include_directories( "$ENV{CUDA_HOME}/include/" )
    link_directories ( "$ENV{CUDA_HOME}/lib64/" )
endif (EXBLAS_GPU_AMD)
set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)

# enabling timing
option (EXBLAS_TIMING "Enable/disable timing of our routines using cpu cycles" OFF)
if (EXBLAS_TIMING)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXBLAS_TIMING")
endif (EXBLAS_TIMING)

# Grab the .c and .cpp files
file (GLOB_RECURSE EXBLAS_C_CPP_SOURCE "common.gpu.cpp" "blas1/*.cpp" "blas2/*.cpp" "blas3/*.cpp" "${PROJECT_SOURCE_DIR}/src/common/*.cpp")
# Grab the .cl files
file (GLOB_RECURSE EXBLAS_CL_SOURCE "blas1/*.cl" "blas2/*.cl" "blas3/*.cl")
# Grab the C/C++ headers
file (GLOB_RECURSE EXBLAS_C_CPP_HEADERS "common.gpu.hpp" "blas1/*.hpp" "blas3/*.hpp" "blas3/*.hpp" "${PROJECT_SOURCE_DIR}/include/*.hpp")
set (EXBLAS_C_CPP_FILES "${EXBLAS_C_CPP_SOURCE};${EXBLAS_C_CPP_HEADERS};${EXBLAS_CL_SOURCE}")

# add the main library
add_library (exblas ${EXBLAS_C_CPP_FILES})
set (EXBLAS_LIB "${PROJECT_BINARY_DIR}/lib")
install (TARGETS exblas DESTINATION ${EXBLAS_LIB})

set (EXBLAS_INCLUDE "${PROJECT_BINARY_DIR}/include")
install (FILES ${EXBLAS_C_CPP_HEADERS} DESTINATION ${EXBLAS_INCLUDE})
install (FILES ${EXBLAS_CL_SOURCE} DESTINATION "${EXBLAS_INCLUDE}/cl")

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

add_subdirectory (blas1)
add_subdirectory (blas2)
add_subdirectory (blas3)

