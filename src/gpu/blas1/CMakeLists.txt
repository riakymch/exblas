# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing ExSUM
add_executable (test.exsum ${PROJECT_SOURCE_DIR}/tests/test.exsum.gpu.cpp)
target_link_libraries (test.exsum ${EXTRA_LIBS})
install (TARGETS test.exsum DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestExSUMNaiveNumbers test.exsum 24)
set_tests_properties (TestExSUMNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExSUMStdDynRange test.exsum 24 2 0 n)
set_tests_properties (TestExSUMStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExSUMLargeDynRange test.exsum 24 50 0 n)
set_tests_properties (TestExSUMLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExSUMIllConditioned test.exsum 24 1e+50 0 i)
set_tests_properties (TestExSUMIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")

# Testing ExDOT
add_executable (test.exdot ${PROJECT_SOURCE_DIR}/tests/test.exdot.gpu.cpp)
target_link_libraries (test.exdot ${EXTRA_LIBS})
install (TARGETS test.exdot DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestExDOTNaiveNumbers test.exdot 24)
set_tests_properties (TestExDOTNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExDOTStdDynRange test.exdot 24 2 0 n)
set_tests_properties (TestExDOTStdDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExDOTLargeDynRange test.exdot 24 50 0 n)
set_tests_properties (TestExDOTLargeDynRange PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExDOTIllConditioned test.exdot 24 1e+50 0 i)
set_tests_properties (TestExDOTIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
