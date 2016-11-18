# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.


# Testing ExGEMV
add_executable (test.exgemv ${PROJECT_SOURCE_DIR}/tests/test.exgemv.gpu.cpp)
target_link_libraries (test.exgemv ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exgemv DESTINATION ${PROJECT_BINARY_DIR}/tests)
add_test (TestExGEMVNaiveNumbers test.exgemv N 256 256)
set_tests_properties (TestExGEMVNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestExGEMVLogUnifDist test.exgemv N 256 256 50 0 n)
set_tests_properties (TestExGEMVLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVFpUnifDist test.exgemv N 256 256 10 0 y)
set_tests_properties (TestExGEMVFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestExGEMVIllConditioned test.exgemv N 256 256 1e+50 0 i)
set_tests_properties (TestExGEMVIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")


# Testing ExTRSV
add_executable (test.extrsv ${PROJECT_SOURCE_DIR}/tests/test.extrsv.gpu.cpp)
add_executable (test.extrsv.exir ${PROJECT_SOURCE_DIR}/tests/test.extrsv.exir.gpu.cpp)
add_executable (test.dtrsv.exir ${PROJECT_SOURCE_DIR}/tests/test.dtrsv.exir.gpu.cpp)
target_link_libraries (test.extrsv ${EXTRA_LIBS})
target_link_libraries (test.extrsv.exir ${EXTRA_LIBS})
target_link_libraries (test.dtrsv.exir ${EXTRA_LIBS})

if (EXBLAS_GPU_NVIDIA)
    install (TARGETS test.extrsv DESTINATION ${PROJECT_BINARY_DIR}/tests)
    add_test (TestExTRSVNaiveNumbers test.extrsv U 256)
    set_tests_properties (TestExTRSVNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    add_test (TestExTRSVLogUnifDist test.extrsv U 256 50 0 n)
    set_tests_properties (TestExTRSVLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
    add_test (TestExTRSVFpUnifDist test.extrsv U 256 10 0 y)
    set_tests_properties (TestExTRSVFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    add_test (TestExTRSVIllConditioned test.extrsv U 256 1e+50 0 i)
    set_tests_properties (TestExTRSVIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")

    install (TARGETS test.extrsv.exir DESTINATION ${PROJECT_BINARY_DIR}/tests)
    #add_test (TestExTRSVExIRNaiveNumbers test.extrsv.exir U 256)
    #set_tests_properties (TestExTRSVExIRNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    #add_test (TestExTRSVExIRLogUnifDist test.extrsv.exir U 256 50 0 n)
    #set_tests_properties (TestExTRSVExIRLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
    #add_test (TestExTRSVExIRFpUnifDist test.extrsv.exir U 256 10 0 y)
    #set_tests_properties (TestExTRSVExIRFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    #add_test (TestExTRSVExIRIllConditioned test.extrsv.exir U 256 1e+50 0 i)
    #set_tests_properties (TestExTRSVExIRIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")

    install (TARGETS test.dtrsv.exir DESTINATION ${PROJECT_BINARY_DIR}/tests)
    #add_test (TestDTRSVExIRNaiveNumbers test.dtrsv.exir U 256)
    #set_tests_properties (TestDTRSVExIRNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    #add_test (TestDTRSVExIRLogUnifDist test.dtrsv.exir U 256 50 0 n)
    #set_tests_properties (TestDTRSVExIRLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
    #add_test (TestDTRSVExIRFpUnifDist test.dtrsv.exir U 256 10 0 y)
    #set_tests_properties (TestDTRSVExIRFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    #add_test (TestDTRSVExIRIllConditioned test.dtrsv.exir U 256 1e+50 0 i)
    #set_tests_properties (TestDTRSVExIRIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
endif (EXBLAS_GPU_NVIDIA)

