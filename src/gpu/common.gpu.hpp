/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

/**
 *  \file  common.gpu.hpp
 *  \brief Provides a set of commong routines for computing on GPUs.
 *         For internal use
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef COMMON_GPU_HPP_
#define COMMON_GPU_HPP_

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif


////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
/**
 * \ingroup ExSUM
 * \brief Function to obtain platform. For internal use
 *
 * \param name Platform name
 * \return Platform ID
 */
cl_platform_id GetOCLPlatform(
    char name[]
);

#if 1
/**
 * \ingroup ExSUM
 * \brief Function to obtain device. For internal use
 * 
 * \param pPlatform Platform ID
 * \return Device ID
 */
cl_device_id GetOCLDevice(
    cl_platform_id pPlatform
);

#else
/**
 * \ingroup ExSUM
 * \brief Function to obtain device by name. For internal use
 *
 * \param pPlatform Platform ID
 * \param name Device name
 * \return Device ID
 */
cl_device_id GetOCLDevice(
    cl_platform_id pPlatform,
    char name[]
);
#endif

#endif // COMMON_GPU_HPP_
