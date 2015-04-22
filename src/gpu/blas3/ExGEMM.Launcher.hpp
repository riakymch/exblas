/*
 *  Copyright (c) 2013-2015 University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file ExGEMM.Launcher.hpp
 *  \brief Provides a set of routines for executing gemm on GPUs
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef ExGEMM_LAUNCHER_H
#define ExGEMM_LAUNCHER_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cassert>
#include <cstring>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////

typedef long long int bintype;

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
/**
 * \ingroup ExGEMM
 * \brief Function to obtain platform. For internal use
 *
 * \param Platform name
 * \return Platform ID
 */
cl_platform_id GetOCLPlatform(
    char name[]
);

/**
 * \ingroup ExGEMM
 * \brief Function to obtain device. For internal use
 * 
 * \param Platform ID
 * \return Device ID
 */
cl_device_id GetOCLDevice(
    cl_platform_id pPlatform
);

/**
 * \ingroup ExGEMM
 * \brief Function to obtain device by name. For internal use
 *
 * \param Platform ID
 * \param Device name
 * \return Device ID
 */
cl_device_id GetOCLDevice(
    cl_platform_id pPlatform,
    char name[]
);


////////////////////////////////////////////////////////////////////////////////
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
/**
 * \ingroup ExGEMM
 * \brief Function to initialize execution on GPUs by allocating kernels and 
 *     memory space. For internal use
 *
 * \param GPU context
 * \param Command queue
 * \param Device ID
 * \param OpenCL file to execute
 * \param Size of FPEs
 * \return Status
 */
extern "C" cl_int initExGEMM(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE
);

/**
 * \ingroup ExGEMM
 * \brief Function to finish the execution on GPUs. It is sort of garbage collector.
 *     For internal use
 */
extern "C" void closeExGEMM(
    void
);


/**
 * \ingroup ExGEMM
 * \brief Executes on GPU parallel matrix-matrix multiplication. For internal use
 *
 * \param Command queue
 * \param m nb of rows of matrix C
 * \param n nb of columns of matrix C
 * \param k nb of rows in matrix B
 * \param alpha scalar
 * \param a matrix A
 * \param lda leading dimension of A
 * \param b matrix B
 * \param ldb leading dimension of B
 * \param beta scalar
 * \param c matrix C
 * \param ldc leading dimension of C
 * \param Error number (output)
 * \return status
 */
extern "C" size_t ExGEMM(
    cl_command_queue cqCommandQueue,
    const uint m,
    const uint n,
    const uint k,
    const double alpha,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_b,
    const uint ldb,
    const double beta,
    cl_mem d_c,
    const uint ldc,
    cl_int *ciErrNumRes
);

#endif
