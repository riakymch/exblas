# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing
add_executable (test.sum ${PROJECT_SOURCE_DIR}/tests/test.sum.cpu.cpp)
target_link_libraries (test.sum ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.sum DESTINATION ${PROJECT_BINARY_DIR}/tests)

if (EXBLAS_MPI)
    add_test (TestSumNaiveNumbers mpirun ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/tests/test.sum 24)
    set_tests_properties (TestSumNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
    add_test (TestSumStdDynRange mpirun ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/tests/test.sum 24 2 0 n)
    set_tests_properties (TestSumStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
    add_test (TestSumLargeDynRange mpirun ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/tests/test.sum 24 50 0 n)
    set_tests_properties (TestSumLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
    add_test (TestSumIllConditioned mpirun ${MPIEXEC_NUMPROC_FLAG} 2 ${PROJECT_BINARY_DIR}/tests/test.sum 24 1e+50 0 i)
    set_tests_properties (TestSumIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
else (EXBLAS_MPI)
    add_test (TestSumNaiveNumbers test.sum 24)
    set_tests_properties (TestSumNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    add_test (TestSumStdDynRange test.sum 24 2 0 n)
    set_tests_properties (TestSumStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    add_test (TestSumLargeDynRange test.sum 24 50 0 n)
    set_tests_properties (TestSumLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
    add_test (TestSumIllConditioned test.sum 24 1e+50 0 i)
    set_tests_properties (TestSumIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
endif (EXBLAS_MPI)

