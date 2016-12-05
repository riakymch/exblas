# Copyright (c) 2016 Inria and University Pierre and Marie Curie
# All rights reserved.

cmake_minimum_required (VERSION 2.8.8)
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
include(CheckFunctionExists)

project (EXBLAS)
enable_language (CXX)

# The version number.
set (EXBLAS_VERSION_MAJOR 1)
set (EXBLAS_VERSION_MINOR 0)
set (EXBLAS_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set (EXBLAS_BINARY_DIR ${PROJECT_BINARY_DIR})

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/cmake/config.h.in"
  "${PROJECT_BINARY_DIR}/include/config.h"
)

# include dir with config.h.in
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")

# enable testing
enable_testing ()

# Recurse into the "src" subdirectory. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.
add_subdirectory (src)

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set (CPACK_PACKAGE_VERSION_MAJOR "${EXBLAS_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${EXBLAS_VERSION_MINOR}")
include (CPack)

# Generate documentation
# add a target to generate API documentation with Doxygen
option (EXBLAS_DOC "Generate the ExBLAS documentation using Doxygen and LaTeX" OFF)
if (EXBLAS_DOC)
    find_package (Doxygen)
    if (DOXYGEN_FOUND)
        configure_file (${PROJECT_SOURCE_DIR}/doc/Doxyfile.in ${PROJECT_BINARY_DIR}/doc/Doxyfile @ONLY)
        add_custom_target (doc
            ${DOXYGEN_EXECUTABLE} ${PROJECT_BINARY_DIR}/doc/Doxyfile
            WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
        # to build latex documentation
        include (tests/UseLATEX)
        SET (LATEX_OUTPUT_PATH ${PROJECT_BINARY_DIR}/doc/latex)
        ADD_LATEX_DOCUMENT (
            refman.tex
            USE_INDEX
            USE_GLOSSARY
        )
        add_dependencies(pdf doc)
    endif (DOXYGEN_FOUND)
endif (EXBLAS_DOC)

