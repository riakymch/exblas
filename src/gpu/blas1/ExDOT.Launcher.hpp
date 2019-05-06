/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file  ExDOT.Launcher.hpp
 *  \brief Provides a set of routines for executing dot product on GPUs.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef EXDOT_LAUNCHER_HPP_
#define EXDOT_LAUNCHER_HPP_

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
 * \ingroup ExDOT
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
extern "C" cl_int initExDOT(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE
);

/**
 * \ingroup ExDOT
 * \brief Function to finish the execution on GPUs. It is sort of garbage collector.
 *     For internal use
 */
extern "C" void closeExDOT(
    void
);

/**
 * \ingroup ExDOT
 * \brief Executes parallel dot product on GPUs. For internal use
 *
 * \param NbElements Nb of elements of two vectors
 * \param d_a vector a
 * \param inca increment of vector a
 * \param offseta from the beginning of vector a
 * \param d_b vector b
 * \param incb increment of vector b
 * \param offsetb from the beginning of vector b
 * \param cqCommandQueue Command queue
 * \param d_Res Result of summation rounded to the nearest
 * \param ciErrNum Error number (output)
 * \return status
 */
extern size_t ExDOT(
    cl_uint NbElements,
    cl_mem d_a,
    const cl_uint inca,
    const cl_uint offseta,
    cl_mem d_b,
    const cl_uint incb,
    const cl_uint offsetb,
    cl_command_queue cqCommandQueue,
    cl_mem d_Res,
    cl_int *ciErrNum
);

/**
 * \ingroup ExDOT
 * \brief Executes parallel dot product on GPUs. For internal use
 *
 * \param NbElements Nb of elements of two vectors
 * \param d_a vector a
 * \param inca increment of vector a
 * \param offseta from the beginning of vector a
 * \param d_b vector b
 * \param incb increment of vector b
 * \param offsetb from the beginning of vector b
 * \param cqCommandQueue Command queue
 * \return result
 */
extern double ExDOT(
    cl_uint NbElements,
    cl_mem d_a,
    const cl_uint inca,
    const cl_uint offseta,
    cl_mem d_b,
    const cl_uint incb,
    const cl_uint offsetb,
    cl_command_queue cqCommandQueue
);

#endif // EXDOT_LAUNCHER_HPP_
