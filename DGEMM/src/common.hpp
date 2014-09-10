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

#include "DGEMM.hpp"
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

typedef struct {
    int width;
    int height;
    int stride;
    cl_mem elements;
} Matrix;

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

cl_device_id GetOCLDevice(
    cl_platform_id pPlatform
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
extern "C" cl_int initDGEMM(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDGEMMVolkov(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDGEMMAMD(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDGEMMNVIDIA(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file
);

extern "C" cl_int initDGEMMNVIDIAGlobal(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE,
    const uint width,
    const uint height,
    const int multi
);

extern "C" cl_int initDGEMMNVIDIAPrivate(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE
);

extern "C" void closeDGEMM(
    void
);

extern "C" void closeDGEMMVolkov(
    void
);

extern "C" void closeDGEMMAMD(
    void
);

extern "C" void closeDGEMMNVIDIA(
    void
);

extern "C" void closeDGEMMNVIDIAGlobal(
    void
);

extern "C" void closeDGEMMNVIDIAPrivate(
    void
);

extern "C" size_t DGEMM(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    const uint n,
    const uint k,
    cl_int *ciErrNum
);

extern "C" size_t DGEMMAMD(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    cl_int *ciErrNum
);

extern "C" size_t DGEMMNVIDIA(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    const uint n,
    cl_int *ciErrNum
);

extern "C" size_t DGEMMNVIDIAGlobal(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    const uint n,
    const int multi,
    cl_int *ciErrNum
);

extern "C" size_t DGEMMNVIDIAPrivate(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    const uint n,
    const int multi,
    cl_int *ciErrNum
);

extern "C" size_t DGEMMVolkov(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    const uint n,
    const uint k,
    cl_int *ciErrNum
);


////////////////////////////////////////////////////////////////////////////////
// Executable functions from main.cpp
////////////////////////////////////////////////////////////////////////////////
int runDGEMM(const char*);

#endif
