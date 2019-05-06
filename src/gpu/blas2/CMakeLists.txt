# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.


# Testing ExGEMV
add_executable (test.exgemv ${PROJECT_SOURCE_DIR}/tests/test.exgemv.gpu.cpp)
target_link_libraries (test.exgemv ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exgemv DESTINATION ${PROJECT_BINARY_DIR}/tests)
# trans = N 	m = n = 512
add_test (TestExGEMVNaiveNumbersN=M test.exgemv N 512 512)
set_tests_properties (TestExGEMVNaiveNumbersN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVLogUnifDistN=M test.exgemv N 512 512 50 0 n)
set_tests_properties (TestExGEMVLogUnifDistN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVFpUnifDistN=M test.exgemv N 512 512 10 0 y)
set_tests_properties (TestExGEMVFpUnifDistN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVIllConditionedN=M test.exgemv N 512 512 1e+50 0 i)
set_tests_properties (TestExGEMVIllConditionedN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
# trans = N 	m = 512		n = 1024
add_test (TestExGEMVNaiveNumbersM<N test.exgemv N 512 1024)
set_tests_properties (TestExGEMVNaiveNumbersM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVLogUnifDistM<N test.exgemv N 512 1024 50 0 n)
set_tests_properties (TestExGEMVLogUnifDistM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVFpUnifDistM<N test.exgemv N 512 1024 10 0 y)
set_tests_properties (TestExGEMVFpUnifDistM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVIllConditionedM<N test.exgemv N 512 1024 1e+50 0 i)
set_tests_properties (TestExGEMVIllConditionedM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
# trans = N 	m = 1024		n = 512
add_test (TestExGEMVNaiveNumbersM>N test.exgemv N 1024 512)
set_tests_properties (TestExGEMVNaiveNumbersM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVLogUnifDistM>N test.exgemv N 1024 512 50 0 n)
set_tests_properties (TestExGEMVLogUnifDistM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVFpUnifDistM>N test.exgemv N 1024 512 10 0 y)
set_tests_properties (TestExGEMVFpUnifDistM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMVIllConditionedM>N test.exgemv N 1024 512 1e+50 0 i)
set_tests_properties (TestExGEMVIllConditionedM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
# trans = T 	m = n = 512
add_test (TestExGEMV^TNaiveNumbersN=M test.exgemv T 512 512)
set_tests_properties (TestExGEMV^TNaiveNumbersN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TLogUnifDistN=M test.exgemv T 512 512 50 0 n)
set_tests_properties (TestExGEMV^TLogUnifDistN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TFpUnifDistN=M test.exgemv T 512 512 10 0 y)
set_tests_properties (TestExGEMV^TFpUnifDistN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TIllConditionedN=M test.exgemv T 512 512 1e+50 0 i)
set_tests_properties (TestExGEMV^TIllConditionedN=M PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
# trans = T 	m = 512		n = 1024
add_test (TestExGEMV^TNaiveNumbersM<N test.exgemv T 512 1024)
set_tests_properties (TestExGEMV^TNaiveNumbersM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TLogUnifDistM<N test.exgemv T 512 1024 50 0 n)
set_tests_properties (TestExGEMV^TLogUnifDistM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TFpUnifDistM<N test.exgemv T 512 1024 10 0 y)
set_tests_properties (TestExGEMV^TFpUnifDistM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TIllConditionedM<N test.exgemv T 512 1024 1e+50 0 i)
set_tests_properties (TestExGEMV^TIllConditionedM<N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
# trans = T 	m = 1024		n = 512
add_test (TestExGEMV^TNaiveNumbersM>N test.exgemv T 1024 512)
set_tests_properties (TestExGEMV^TNaiveNumbersM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TLogUnifDistM>N test.exgemv T 1024 512 50 0 n)
set_tests_properties (TestExGEMV^TLogUnifDistM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TFpUnifDistM>N test.exgemv T 1024 512 10 0 y)
set_tests_properties (TestExGEMV^TFpUnifDistM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMV^TIllConditionedM>N test.exgemv T 1024 512 1e+50 0 i)
set_tests_properties (TestExGEMV^TIllConditionedM>N PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")


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

