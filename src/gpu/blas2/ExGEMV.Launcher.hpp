/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file  ExGEMV.Launcher.hpp
 *  \brief Provides a set of routines for executing gemv on GPUs.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXGEMV_LAUNCHER_HPP_
#define EXGEMV_LAUNCHER_HPP_

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
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
/**
 * \ingroup ExGEMV
 * \brief Function to initialize execution on GPUs by allocating kernels and 
 *     memory space. For internal use
 *
 * \param cxGPUContext GPU context
 * \param cqParamCommandQue Command queue
 * \param cdDevice Device ID
 * \param program_file OpenCL file to execute
 * \param m nb of rows of matrix A
 * \param NbFPE Size of FPEs
 * \return Status
 */
extern "C" cl_int initExGEMV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint m,
    const uint NbFPE
);

/**
 * \ingroup ExGEMV
 * \brief Function to finish the execution on GPUs. It is sort of garbage collector.
 *     For internal use
 */
extern "C" void closeExGEMV(
    void
);

/**
 * \ingroup ExGEMV
 * \brief Executes parallel matrix-vector multiplication on GPU. For internal use
 *
 * \param cqCommandQueue Command queue
 * \param m nb of rows of matrix A
 * \param n nb of columns of matrix A
 * \param alpha scalar
 * \param d_a matrix A
 * \param lda leading dimension of A
 * \param d_x vector
 * \param incx the increment for the elements of a
 * \param beta scalar
 * \param d_y vector
 * \param incy the increment for the elements of a
 * \param ciErrNum Error number (output)
 * \return status
 */
extern "C" size_t ExGEMV(
    cl_command_queue cqCommandQueue,
    const uint m,
    const uint n,
    const double alpha,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_x,
    const uint incx,
    const double beta,
    const cl_mem d_y,
    const uint incy,
    cl_int *ciErrNum
);

#endif // EXGEMV_LAUNCHER_HPP_

