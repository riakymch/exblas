/*
 *  Copyright (c) 2013-2015 University Pierre and Marie Curie 
 *  All rights reserved.
 */

#ifndef ExGEMM_LAUNCHER_H
#define ExGEMM_LAUNCHER_H

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


////////////////////////////////////////////////////////////////////////////////
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExGEMM(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint mi,
    const uint ni,
    const uint ki,
    const uint NbFPE
);

extern "C" void closeExGEMM(
    void
);

extern "C" size_t runExGEMM(
    cl_command_queue cqCommandQueue,
    const uint m,
    const uint n,
    const uint k,
    const double alpha,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_b,
    const uint ldb,
    const double beta,
    cl_mem d_c,
    const uint ldc,
    cl_int *ciErrNumRes
);

#endif
