# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing
add_executable (test.exgemm ${PROJECT_SOURCE_DIR}/tests/test.exgemm.gpu.cpp)
target_link_libraries (test.exgemm ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exgemm DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestSumNaiveNumbers test.exgemm 512 512 512)
set_tests_properties (TestSumNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestSumStdDynRange test.exgemm 512 512 512 2 0 n)
set_tests_properties (TestSumStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestSumLargeDynRange test.exgemm 512 512 512 50 0 n)
set_tests_properties (TestSumLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestSumIllConditioned test.exgemm 512 512 512 1e+50 0 i)
set_tests_properties (TestSumIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
