/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file gpu/blas3/ExGEMM.Launcher.hpp
 *  \brief Provides a set of routines for executing gemm on GPUs.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXGEMM_LAUNCHER_HPP_
#define EXGEMM_LAUNCHER_HPP_

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
 * \ingroup ExGEMM
 * \brief Function to initialize execution on GPUs by allocating kernels and 
 *     memory space. For internal use
 *
 * \param cxGPUContext GPU context
 * \param cqParamCommandQue Command queue
 * \param cdDevice Device ID
 * \param program_file OpenCL file to execute
 * \param NbFPE Size of FPEs
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
 * \brief Executes parallel matrix-matrix multiplication on GPU. For internal use
 *
 * \param cqCommandQueue Command queue
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
 * \param ciErrNumRes Error number (output)
 * \return status
 */
extern "C" size_t ExGEMM(
    cl_command_queue cqCommandQueue,
    const uint m,
    const uint n,
    const uint k,
    const double alpha,
    const cl_mem a,
    const uint lda,
    const cl_mem b,
    const uint ldb,
    const double beta,
    cl_mem c,
    const uint ldc,
    cl_int *ciErrNumRes
);

#endif //EXGEMM_LAUNCHER_HPP_

