# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.


# Testing ExGEMV
add_executable (test.exgemv ${PROJECT_SOURCE_DIR}/tests/test.exgemv.gpu.cpp)
target_link_libraries (test.exgemv ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exgemv DESTINATION ${PROJECT_BINARY_DIR}/tests)
add_test (TestExGEMVNaiveNumbers test.exgemv N 256 256)
set_tests_properties (TestExGEMVNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVLogUnifDist test.exgemv N 256 256 50 0 n)
set_tests_properties (TestExGEMVLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVFpUnifDist test.exgemv N 256 256 10 0 y)
set_tests_properties (TestExGEMVFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVIllConditioned test.exgemv N 256 256 1e+50 0 i)
set_tests_properties (TestExGEMVIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")


# Testing ExTRSV
add_executable (test.extrsv ${PROJECT_SOURCE_DIR}/tests/test.extrsv.gpu.cpp)
target_link_libraries (test.extrsv ${EXTRA_LIBS})

if (EXBLAS_GPU_NVIDIA)
    install (TARGETS test.extrsv DESTINATION ${PROJECT_BINARY_DIR}/tests)
    add_test (TestExTRSVNaiveNumbers test.extrsv U N N 256)
    set_tests_properties (TestExTRSVNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
    add_test (TestExTRSVLogUnifDist test.extrsv U N N 256 50 0 n)
    set_tests_properties (TestExTRSVLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
    add_test (TestExTRSVFpUnifDist test.extrsv U N N 256 10 0 y)
    set_tests_properties (TestExTRSVFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
    add_test (TestExTRSVIllConditioned test.extrsv U N N 256 1e+50 0 i)
    set_tests_properties (TestExTRSVIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
endif (EXBLAS_GPU_NVIDIA)

