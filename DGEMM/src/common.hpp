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

typedef long long int bintype;

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

void print2Superaccumulators(
    bintype *binCPU,
    bintype *binGPU
);

////////////////////////////////////////////////////////////////////////////////
// CPU superaccumulator
////////////////////////////////////////////////////////////////////////////////
extern "C" double roundSuperaccumulator(
    bintype *bin
);

////////////////////////////////////////////////////////////////////////////////
// GPU superaccumulator related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initSuperaccumulator(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    uint NbFPE
);

extern "C" void closeSuperaccumulator(
    void
);

extern "C" size_t Superaccumulate(
    cl_command_queue cqCommandQueue,
    cl_mem d_Accumulator,
    cl_mem d_Data,
    uint nbElements,
    cl_int *ciErrNum
);

////////////////////////////////////////////////////////////////////////////////
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initReduction(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" void closeReduction(
    void
);

extern "C" size_t Reduction(
    cl_command_queue cqCommandQueue,
    cl_mem d_oData,
    cl_mem d_iData,
    uint nbElements,
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

// extern "C" void KahanSummation(double *s, double *c, double d);
// extern "C" double roundKulish(bintype *bin);
extern "C" double roundKahan(
    double *data,
    int size
);


////////////////////////////////////////////////////////////////////////////////
// Executable functions from main.cpp
////////////////////////////////////////////////////////////////////////////////
int runSuperaccumulator(const char*);
int runReduction(const char*);

#endif
