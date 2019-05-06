# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing
add_executable (test.exsum ${PROJECT_SOURCE_DIR}/tests/test.exsum.mic.cpp)
target_link_libraries (test.exsum ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.exsum DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestSumNaiveNumbers micnativeloadex ${PROJECT_BINARY_DIR}/tests/test.exsum -a "24 1")
set_tests_properties (TestSumNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
add_test (TestSumStdDynRange micnativeloadex
${PROJECT_BINARY_DIR}/tests/test.exsum -a "24 2 0 n")
set_tests_properties (TestSumStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
add_test (TestSumLargeDynRange micnativeloadex
${PROJECT_BINARY_DIR}/tests/test.exsum -a "24 50 0 n")
set_tests_properties (TestSumLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")
add_test (TestSumIllConditioned micnativeloadex
${PROJECT_BINARY_DIR}/tests/test.exsum -a "24 1e+50 0 i")
set_tests_properties (TestSumIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; All OK!")

