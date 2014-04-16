/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#include <gmp.h>
#include <mpfr.h>

#include "Superaccumulator.hpp"

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define BIN_COUNT 39U


#define E_BITS 1023
#define F_BITS (1023 + 52)

typedef cl_long bintype;


////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
cl_platform_id GetOCLPlatform(
    char name[]
);

cl_device_id GetOCLDevice(
    cl_platform_id pPlatform,
    char name[]
);

void initVectorWithRandomData(
    double *data,
    int size
);

void init_fpuniform(
    double *array,
    int size,
    int range,
    int emax
);

double min(
    double arr[],
    int size
);

////////////////////////////////////////////////////////////////////////////////
// GPU DDOT product related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initDDOT(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDDOTAMD(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDDOTNVIDIA(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDDOTRepro(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE,
    const uint width,
    const uint height
);

extern "C" void closeDDOT(
    void
);

extern "C" void closeDDOTAMD(
    void
);

extern "C" void closeDDOTNVIDIA(
    void
);

extern "C" void closeDDOTRepro(
    void
);

extern "C" size_t DDOT(
    cl_command_queue cqCommandQueue,
    cl_mem d_res,
    const cl_mem d_a,
    const cl_mem d_b,
    const uint NbElements,
    cl_int *ciErrNum
);

extern "C" size_t DDOTAMD(
    cl_command_queue cqCommandQueue,
    cl_mem d_res,
    const cl_mem d_a,
    const cl_mem d_b,
    const uint NbElements,
    cl_int *ciErrNum
);

extern "C" size_t DDOTNVIDIA(
    cl_command_queue cqCommandQueue,
    cl_mem d_res,
    const cl_mem d_a,
    const cl_mem d_b,
    const uint NbElements,
    cl_int *ciErrNum
);

extern "C" size_t DDOTRepro(
    cl_command_queue cqCommandQueue,
    cl_mem d_res,
    const cl_mem d_a,
    const cl_mem d_b,
    const uint NbElements,
    cl_int *ciErrNum
);

////////////////////////////////////////////////////////////////////////////////
// MPFR and Kahan summation functions
////////////////////////////////////////////////////////////////////////////////
extern "C" char *sum_mpfr(
    double *data,
    int size);

extern "C" double round_mpfr(
    double *data,
    int size
);


////////////////////////////////////////////////////////////////////////////////
// Executable functions from main.cpp
////////////////////////////////////////////////////////////////////////////////
int runDDOT(const char*);

#endif
