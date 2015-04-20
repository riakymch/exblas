find_package(MPI)
if(NOT MPI_C_FOUND)
  message(FATAL_ERROR "MPI C compiler was not found and is required")
endif()
if(NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "MPI C++ compiler was not found and is required")
endif()
include_directories(${MPI_CXX_INCLUDE_PATH})
set(EXTRA_FLAGS "${EXTRA_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "${MPI_CXX_COMPILE_FLAGS} ${MPI_LINK_FLAGS}")
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_REQUIRED_LIBRARIES ${MPI_CXX_LIBRARIES})

# Ensure that we have MPI_[UNSIGNED_]LONG_LONG if compiling with 64-bit integers
# ==============================================================================
set(MPI_LONG_LONG_CODE
    "#include \"mpi.h\"
     int main( int argc, char* argv[] )
     {
         MPI_Init( &argc, &argv );
         MPI_Datatype lli = MPI_LONG_LONG_INT;
         MPI_Datatype llu = MPI_UNSIGNED_LONG_LONG;
         MPI_Finalize();
         return 0;
     }")
check_cxx_source_compiles("${MPI_LONG_LONG_CODE}" HAVE_MPI_LONG_LONG)
if(USE_64BIT_INTS AND NOT HAVE_MPI_LONG_LONG)
  message(FATAL_ERROR 
    "Did not detect MPI_LONG_LONG_INT and MPI_UNSIGNED_LONG_LONG")
endif()

