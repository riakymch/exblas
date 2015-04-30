# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing
add_executable (test.exsum ${PROJECT_SOURCE_DIR}/tests/test.exsum.gpu.cpp)
target_link_libraries (test.exsum ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exsum DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestExSUMNaiveNumbers test.exsum 24)
set_tests_properties (TestExSUMNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestExSUMStdDynRange test.exsum 24 2 0 n)
set_tests_properties (TestExSUMStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExSUMLargeDynRange test.exsum 24 50 0 n)
set_tests_properties (TestExSUMLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestExSUMIllConditioned test.exsum 24 1e+50 0 i)
set_tests_properties (TestExSUMIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
