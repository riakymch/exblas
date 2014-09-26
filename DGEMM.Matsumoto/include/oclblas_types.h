#ifndef OCLBLAS_TYPES_H_
#define OCLBLAS_TYPES_H_

#include <assert.h>
#include <sys/types.h>
#include <CL/cl.h>

typedef cl_int oclblas_err_t;

#define OCLBLAS_TRUE  (1)
#define OCLBLAS_FALSE (0)

// Error values
#define OCLBLAS_SUCCESS                    CL_SUCCESS                    // 0
#define OCLBLAS_DEVICE_NOT_FOUND           CL_DEVICE_NOT_FOUND           // -1
#define OCLBLAS_DEVICE_NOT_AVAILABLE       CL_DEVICE_NOT_AVAILABLE       // -2
#define OCLBLAS_COMPILER_NOT_AVAILABLE     CL_COMPILER_NOT_AVAILABLE     // -3
#define OCLBLAS_OUT_OF_HOST_MEMORY         CL_OUT_OF_HOST_MEMORY         // -6
#define OCLBLAS_BUILD_PROGRAM_FAILURE      CL_BUILD_PROGRAM_FAILURE      // -11
#define OCLBLAS_INVALID_VALUE              CL_INVALID_VALUE              // -30
#define OCLBLAS_INVALID_DEVICE_TYPE        CL_INVALID_DEVICE_TYPE        // -31
#define OCLBLAS_INVALID_PLATFORM           CL_INVALID_PLATFORM           // -32
#define OCLBLAS_INVALID_DEVICE             CL_INVALID_DEVICE             // -33
#define OCLBLAS_INVALID_CONTEXT            CL_INVALID_CONTEXT            // -34
#define OCLBLAS_INVALID_QUEUE_PROPERTIES   CL_INVALID_QUEUE_PROPERTIES   // -35
#define OCLBLAS_INVALID_MEM_OBJECT         CL_INVALID_MEM_OBJECT         // -38
#define OCLBLAS_INVALID_BINARY             CL_INVALID_BINARY             // -42
#define OCLBLAS_INVALID_BUILD_OPTIONS      CL_INVALID_BUILD_OPTIONS      // -43
#define OCLBLAS_INVALID_PROGRAM            CL_INVALID_PROGRAM            // -44
#define OCLBLAS_INVALID_PROGRAM_EXECUTABLE CL_INVALID_PROGRAM_EXECUTABLE // -45
#define OCLBLAS_INVALID_KERNEL_NAME        CL_INVALID_KERNEL_NAME        // -46
#define OCLBLAS_INVALID_KERNEL_DEFINITION  CL_INVALID_KERNEL_DEFINITION  // -47
#define OCLBLAS_INVALID_KERNEL             CL_INVALID_KERNEL             // -48
#define OCLBLAS_INVALID_ARG_INDEX          CL_INVALID_ARG_INDEX          // -49
#define OCLBLAS_INVALID_ARG_VALUE          CL_INVALID_ARG_VALUE          // -50
#define OCLBLAS_INVALID_OPERATION          CL_INVALID_OPERATION          // -59
#define OCLBLAS_INVALID_BUFFER_SIZE        CL_INVALID_BUFFER_SIZE        // -61
#define OCLBLAS_UNKNOWN_ERROR              -1000
#define OCLBLAS_RT_ALREADY_INITIALIZED     -1001
#define OCLBLAS_RT_NOT_INITIALIZED         -1002
#define OCLBLAS_OPEN_FILE_FAILURE          -1003
#define OCLBLAS_READ_FILE_FAILURE          -1004
#define OCLBLAS_INVALID_FUNCTION_CALL      -1005
#define OCLBLAS_PROGRAM_NOT_FIND           -1006
#define OCLBLAS_INVALID_FUNC_ARGUMENT      -1007
#define OCLBLAS_CREATE_BUFFER_FAILURE      -1008

// Parameter constants
typedef int oclblas_order_t;
#define OclblasRowMajor       101
#define OclblasColMajor       102

typedef int oclblas_trans_t;
#define OclblasNoTrans        111
#define OclblasTrans          112
#define OclblasConjTrans      113

typedef int oclblas_uplo_t;
#define OclblasUpper          121
#define OclblasLower          122
#define OclblasFull           123

typedef int oclblas_diag_t;
#define OclblasNonUnit        131
#define OclblasUnit           132

typedef int oclblas_side_t;
#define OclblasLeft           141
#define OclblasRight          142

typedef int oclblas_vec_t;

#if defined(TARGET_OPENCL_PLATFORM_IDX)
#define OCLBLAS_DEFAULT_PLATFORM TARGET_OPENCL_PLATFORM_IDX
#else
#define OCLBLAS_DEFAULT_PLATFORM 0
#endif

#if defined(TARGET_OPENCL_DEVICE_GPU)
#define OCLBLAS_DEFAULT_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#elif defined(TARGET_OPENCL_DEVICE_CPU)
#define OCLBLAS_DEFAULT_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#elif defined(USE_OPENCL_ACCELERATOR_DEVICE)
#define OCLBLAS_DEFAULT_DEVICE_TYPE CL_DEVICE_TYPE_ACCELERATOR
#else // Default
#define OCLBLAS_DEFAULT_DEVICE_TYPE CL_DEVICE_TYPE_DEFAULT
#endif

#endif // #ifndef OCLBLAS_TYPES_H_
