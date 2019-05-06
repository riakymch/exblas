/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
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
 * \param transa transpose ('T') or non-transpose ('N') matrix A
 * \param m nb of rows of matrix A
 * \param imt nb of threads in row, wg size
 * \param p nb of columns in the resulting matrix
 * \param NbFPE Size of FPEs
 * \return Status
 */
extern "C" cl_int initExGEMV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
	const char transa,
    const uint m,
    const uint ip,
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
 * \param transa transpose ('T') or non-transpose ('N') matrix A
 * \param m nb of rows of matrix A
 * \param n nb of columns of matrix A
 * \param alpha scalar
 * \param d_a matrix A
 * \param lda leading dimension of A
 * \param offseta from the beginning of the matrix A
 * \param d_x vector
 * \param incx the increment for the elements of a
 * \param offsetx from the beginning of the vector x
 * \param beta scalar
 * \param d_y vector
 * \param incy the increment for the elements of a
 * \param offsety from the beginning of the vector y
 * \param ciErrNum Error number (output)
 * \return status
 */
extern "C" size_t ExGEMV(
    cl_command_queue cqCommandQueue,
	const char transa,
    const uint m,
    const uint n,
    const double alpha,
    const cl_mem d_a,
    const uint lda,
    const uint offseta,
    const cl_mem d_x,
    const uint incx,
    const uint offsetx,
    const double beta,
    cl_mem d_y,
    const uint incy,
    const uint offsety,
    cl_int *ciErrNum
);

#endif // EXGEMV_LAUNCHER_HPP_

