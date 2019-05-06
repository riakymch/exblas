# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing
add_executable (test.exgemm ${PROJECT_SOURCE_DIR}/tests/test.exgemm.gpu.cpp)
target_link_libraries (test.exgemm ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exgemm DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestExGEMMNaiveNumbers test.exgemm 256 256 256)
set_tests_properties (TestExGEMMNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMMStdDynRange test.exgemm 256 256 256 2 0 n)
set_tests_properties (TestExGEMMStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMMLargeDynRange test.exgemm 256 256 256 50 0 n)
set_tests_properties (TestExGEMMLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExGEMMIllConditioned test.exgemm 256 256 256 1e+50 0 i)
set_tests_properties (TestExGEMMIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
