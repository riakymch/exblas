# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Generate GPU specific library relying upon OpenCL

# add the EXBLAS library?
option (USE_EXBLAS "Use ExBLAS provided reproducible and accurate implementations of the BLAS routines" ON)
if (USE_EXBLAS)
  include_directories ("${PROJECT_SOURCE_DIR}/include")
  include_directories ("${PROJECT_SOURCE_DIR}/src/common")
  include_directories ("${PROJECT_BINARY_DIR}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} exblas)
endif (USE_EXBLAS)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -fabi-version=0 -O3 -Wall")

include (tests/OpenCL)
# architecture specific flags
option (EXBLAS_AMD_GPU "By setting this option we enable compilation for AMD GPUs" OFF)
option (EXBLAS_NVIDIA_GPU "By setting this option we enable compilation for NVIDIA GPUs" OFF)
if (EXBLAS_AMD_GPU)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=att -DAMD")
    include_directories( "/opt/sdks/amd/current/include/" )
    link_directories ( "/opt/sdks/amd/current/lib/x86_64/" )
elseif (EXBLAS_NVIDIA_GPU)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel")
    include_directories( "/usr/local/cuda/include/" )
    link_directories ( "/usr/lib/" )
    link_directories ( "/usr/local/cuda/lib64/" )
endif (EXBLAS_AMD_GPU)
#include_directories( ${OPENCL_INCLUDE_DIRS} )
#link_directories ( ${OpenCL_LIBRARY} )
set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)

# enabling timing
option (EXBLAS_TIMING "Enable/disable timing of our routines using cpu cycles" OFF)
if (EXBLAS_TIMING)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXBLAS_TIMING")
endif (EXBLAS_TIMING)

# Grab the .c and .cpp files
file (GLOB_RECURSE EXBLAS_C_CPP_SOURCE "*.cpp" "${PROJECT_SOURCE_DIR}/src/common/*.cpp")
# Grab the .cl files
file (GLOB_RECURSE EXBLAS_CL_SOURCE "*.cl")
# Grab the C/C++ headers
file (GLOB_RECURSE EXBLAS_C_CPP_HEADERS "${PROJECT_SOURCE_DIR}/include/*.hpp")
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

# Testing
add_executable (test.sum ${PROJECT_SOURCE_DIR}/tests/test.sum.gpu.cpp)
target_link_libraries (test.sum ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.sum DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestSumNaiveNumbers test.sum 24)
set_tests_properties (TestSumNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestSumStdDynRange test.sum 24 2 0 n)
set_tests_properties (TestSumStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestSumLargeDynRange test.sum 24 50 0 n)
set_tests_properties (TestSumLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestSumIllConditioned test.sum 24 1e+50 0 i)
set_tests_properties (TestSumIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")

