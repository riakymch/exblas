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

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string.h>

#include "TRSV.hpp"

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define BIN_COUNT 76U


#define E_BITS 2 * 1023
#define F_BITS 2 * (1023 + 52)

typedef cl_long bintype;


////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
cl_platform_id GetOCLPlatform(
    char name[]
);

cl_device_id GetOCLDevice(
    cl_platform_id pPlatform
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
    double *a,
    const uint n,
    const int range,
    const int emax
);

void init_fpuniform_lu_matrix(
    double *a,
    const uint n,
    const int range,
    const int emax
);

void init_fpuniform_un_matrix(
    double *a,
    const uint n,
    const int range,
    const int emax
);

extern "C" void generate_ill_cond_system(
    int islower,
    double *a,
    double *b,
    uint n,
    const double c
);

double min(
    double arr[],
    int size
);


////////////////////////////////////////////////////////////////////////////////
// Kahan summation functions
////////////////////////////////////////////////////////////////////////////////
extern "C" double TwoProductFMA(
    double a,
    double b,
    double *d
);

extern "C" double KnuthTwoSum(
    double a,
    double b,
    double *d
);

////////////////////////////////////////////////////////////////////////////////
// Executable functions from main.cpp
////////////////////////////////////////////////////////////////////////////////
int runTRSV(const char*);

#endif
