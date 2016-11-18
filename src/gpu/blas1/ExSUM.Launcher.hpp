/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file  ExSUM.Launcher.hpp
 *  \brief Provides a set of routines for executing summation on GPUs.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXSUM_LAUNCHER_HPP_
#define EXSUM_LAUNCHER_HPP_

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
 * \ingroup ExSUM
 * \brief Function to initialize execution on GPUs by allocating kernels and 
 *     memory space. For internal use
 *
 * \param cxGPUContext GPU context
 * \param cqParamCommandQue Command queue
 * \param cdDevice Device ID
 * \param program_file OpenCL file to execute
 * \param NbElements Nb of elements to sum
 * \param NbFPE Size of FPEs
 * \return Status
 */
extern "C" cl_int initExSUM(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbElements,
    const uint NbFPE
);

/**
 * \ingroup ExSUM
 * \brief Function to finish the execution on GPUs. It is sort of garbage collector.
 *     For internal use
 */
extern "C" void closeExSUM(
    void
);

/**
 * \ingroup ExSUM
 * \brief Executes parallel reduction on GPUs. For internal use
 *
 * \param cqCommandQueue Command queue
 * \param d_Res Result of summation rounded to the nearest
 * \param d_a vector to sum up
 * \param inca incremenet of vector a
 * \param offset from the beginning of vector a
 * \param ciErrNum Error number (output)
 * \return status
 */
extern "C" size_t ExSUM(
    cl_command_queue cqCommandQueue,
    cl_mem d_Res,
    cl_mem d_a,
    const cl_uint inca,
    const cl_uint offset,
    cl_int *ciErrNum
);

#endif // EXSUM_LAUNCHER_HPP_
