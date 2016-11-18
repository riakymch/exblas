/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file  ExTRSV.Launcher.hpp
 *  \brief Provides a set of routines for executing trsv on GPUs.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXTRSV_LAUNCHER_HPP_
#define EXTRSV_LAUNCHER_HPP_

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
 * \ingroup ExTRSV
 * \brief Function to initialize execution on GPUs by allocating kernels and 
 *     memory space. For internal use
 *
 * \param cxGPUContext GPU context
 * \param cqParamCommandQue Command queue
 * \param cdDevice Device ID
 * \param program_file OpenCL file to execute
 * \param n size of matrix A
 * \param NbFPE Size of FPEs
 * \return Status
 */
extern "C" cl_int initExTRSV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint n,
    const uint NbFPE
);

/**
 * \ingroup ExTRSV
 * \brief Function to finish the execution on GPUs. It is sort of garbage collector.
 *     For internal use
 */
extern "C" void closeExTRSV(
    void
);

/**
 * \ingroup ExTRSV
 * \brief Executes parallel triangular solver (ExTRSV) on GPU. For internal use
 *
 * \param cqCommandQueue Command queue
 * \param n size of matrix A
 * \param d_a matrix A
 * \param lda leading dimension of A
 * \param offseta from the beginning of the matrix A
 * \param d_x vector
 * \param incx the increment for the elements of a
 * \param offsetx from the beginning of the vector x
 * \param ciErrNum Error number (output)
 * \return status
 */
extern "C" size_t ExTRSV(
    cl_command_queue cqCommandQueue,
    const uint n,
    const cl_mem d_a,
    const uint lda,
    const uint offseta,
    const cl_mem d_x,
    const uint incx,
    const uint offsetx,
    cl_int *ciErrNum
);

/**
 * \ingroup ExTRSV
 * \brief Executes parallel triangular solver (ExTRSV) on GPU with one iteration of the iterative refinement (ExIR).
 *   For internal use
 *
 * \param cqCommandQueue Command queue
 * \param n size of matrix A
 * \param d_a matrix A
 * \param lda leading dimension of A
 * \param d_x vector
 * \param incx the increment for the elements of a
 * \param d_b vector
 * \param ciErrNum Error number (output)
 * \return status
 */
extern "C" size_t ExTRSVIR(
    cl_command_queue cqCommandQueue,
    const uint n,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_x,
    const uint incx,
    const cl_mem d_b,
    cl_int *ciErrNum
);

/**
 * \ingroup ExTRSV
 * \brief Executes parallel triangular solver (DTRSV) on GPU with one iteration of the iterative refinement (ExIR).
 *   For internal use
 *
 * \param cqCommandQueue Command queue
 * \param n size of matrix A
 * \param d_a matrix A
 * \param lda leading dimension of A
 * \param d_x vector
 * \param incx the increment for the elements of a
 * \param d_b vector
 * \param ciErrNum Error number (output)
 * \return status
 */
extern "C" size_t DTRSVExIR(
    cl_command_queue cqCommandQueue,
    const uint n,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_x,
    const uint incx,
    const cl_mem d_b,
    cl_int *ciErrNum
);

#endif // EXTRSV_LAUNCHER_HPP_

