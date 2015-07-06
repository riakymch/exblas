# Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
# All rights reserved.

# Testing
add_executable (test.extrsv ${PROJECT_SOURCE_DIR}/tests/test.extrsv.gpu.cpp)
target_link_libraries (test.extrsv ${EXTRA_LIBS})

# add the install targets
install (TARGETS test.extrsv DESTINATION ${PROJECT_BINARY_DIR}/tests)

add_test (TestExTRSVNaiveNumbers test.extrsv U 256)
set_tests_properties (TestExTRSVNaiveNumbers PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestExTRSVLogUnifDist test.extrsv U 256 50 0 n)
set_tests_properties (TestExTRSVLogUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK")
add_test (TestExTRSVFpUnifDist test.extrsv U 256 10 0 y)
set_tests_properties (TestExTRSVFpUnifDist PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
add_test (TestExTRSVIllConditioned test.extrsv U 256 1e+50 0 i)
set_tests_properties (TestExTRSVIllConditioned PROPERTIES PASS_REGULAR_EXPRESSION "TestPassed; ALL OK!")
