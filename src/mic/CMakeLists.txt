# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Generate CPU specific library using Intel TBB and OpenMP

# add the EXBLAS library?
option (USE_EXBLAS "Use ExBLAS provided reproducible and accurate implementations of the BLAS routines" ON)
if (USE_EXBLAS)
  include_directories ("${PROJECT_SOURCE_DIR}/include")
  include_directories ("${PROJECT_SOURCE_DIR}/src/common")
  include_directories ("${PROJECT_BINARY_DIR}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} exblas)
endif (USE_EXBLAS)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -Wall -mmic -DATT_SYNTAX -DTHREADSAFE -fp-model precise -openmp -openmp-report=1")

# enabling timing
option (EXBLAS_TIMING "Enable/disable timing of our routines using cycles" OFF)
if (EXBLAS_TIMING)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXBLAS_TIMING")
endif (EXBLAS_TIMING)

# add TBB
set (EXTRA_LIBS ${EXTRA_LIBS} tbb)
set (EXTRA_LIBS ${EXTRA_LIBS} pthread)

# Grab the .c and .cpp files
file (GLOB_RECURSE EXBLAS_C_CPP_SOURCE "${PROJECT_SOURCE_DIR}/src/common/*.cpp" "${PROJECT_SOURCE_DIR}/src/common/*.h" "*.hpp" "*.cpp")
# Grab the C/C++ headers
file (GLOB_RECURSE EXBLAS_C_CPP_HEADERS "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.hpp")
set (EXBLAS_C_CPP_FILES "${EXBLAS_C_CPP_SOURCE};${EXBLAS_C_CPP_HEADERS}")

# add the main library
add_library (exblas ${EXBLAS_C_CPP_FILES})
set (EXBLAS_LIB "${PROJECT_BINARY_DIR}/lib")
install (TARGETS exblas DESTINATION ${EXBLAS_LIB})

set (EXBLAS_INCLUDE "${PROJECT_BINARY_DIR}/include")
install (FILES ${EXBLAS_C_CPP_HEADERS} DESTINATION ${EXBLAS_INCLUDE})

# enabling comparing with MPFR
option (EXBLAS_VS_MPFR "Use MPFR to compare the achived results for tests" OFF)
if (EXBLAS_VS_MPFR)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXBLAS_VS_MPFR")
    include_directories ("$ENV{GMP_HOME}/include/")
    include_directories ("$ENV{MPFR_HOME}/include/")
    link_directories ("$ENV{GMP_HOME}/lib/")
    link_directories ("$ENV{MPFR_HOME}/lib/")
    set (EXTRA_LIBS ${EXTRA_LIBS} gmp)
    set (EXTRA_LIBS ${EXTRA_LIBS} mpfr)
endif (EXBLAS_VS_MPFR)

add_subdirectory (blas1)

