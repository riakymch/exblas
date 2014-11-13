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

#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define TRSV_INIT "trsv_init"
#define TRSV_KERNEL "trsv_lnn"
#ifdef AMD
  #define THREADSX 32
  #define THREADSY  4
#else
  #define THREADSX 32
  #define THREADSY  1
#endif

static size_t szKernelLength;                  //Byte size of kernel code
static char*  cSources = NULL;                 //Buffer to hold source for compilation

static cl_program       cpProgram;             //OpenCL program
static cl_kernel        ckInit, ckKernel;
static cl_command_queue cqDefaultCommandQue;   //Default command queue
static cl_mem           d_sync;
static cl_mem           d_Superaccs;
static uint             __alg;

#ifdef AMD
static char  compileOptions[256] = "-DBLOCK_SIZE=32";
#else
static char  compileOptions[256] = "-DNVIDIA -DUSE_KNUTH -DBLOCK_SIZE=32 -Dthreadsx=32 -Dthreadsy=1 -cl-mad-enable -cl-fast-relaxed-math"; // -cl-nv-verbose";
#endif


extern "C" cl_int initTRSV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint n,
    const uint alg,
    const uint NbFPE
){
    cl_int ciErrNum;

    // Read the OpenCL kernel in from source file
    FILE *program_handle;
    printf("Load the program sources (%s)...\n", program_file);
    program_handle = fopen(program_file, "r");
    if (!program_handle) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    szKernelLength = ftell(program_handle);
    rewind(program_handle);
    cSources = (char *) malloc(szKernelLength + 1);
    cSources[szKernelLength] = '\0';
    ciErrNum = fread(cSources, sizeof(char), szKernelLength, program_handle);
    fclose(program_handle);

    printf("clCreateProgramWithSource...\n"); 
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSources, &szKernelLength, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    printf("...building program\n");
        sprintf(compileOptions, "%s -DNBFPE=%d", compileOptions, NbFPE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            //printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[4096];
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }

    printf("...creating kernel(s):\n");
        ckInit = clCreateKernel(cpProgram, TRSV_INIT, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: trsv_init, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckKernel = clCreateKernel(cpProgram, TRSV_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: trsv_lnn, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    printf("...allocating internal buffer\n");
        if (alg) {
            d_Superaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, n * BIN_COUNT * sizeof(cl_long), NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }
        d_sync = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, 2 * sizeof(cl_int), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //Save default command queue and algorithm's number
    cqDefaultCommandQue = cqParamCommandQue;
    __alg = alg;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeTRSV(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseMemObject(d_sync);
    if (__alg)
        ciErrNum  = clReleaseMemObject(d_Superaccs);
    ciErrNum &= clReleaseKernel(ckInit);
    ciErrNum &= clReleaseKernel(ckKernel);
    ciErrNum &= clReleaseProgram(cpProgram);

    if (ciErrNum != CL_SUCCESS)
        printf("Error in closeTRSV(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Reduction / mergeReduction kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t TRSV(
    cl_command_queue cqCommandQueue,
    cl_mem d_x,
    const cl_mem d_a,
    const cl_mem d_b,
    const uint n,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        size_t NbThreadsPerWorkGroup = 1;
        size_t TotalNbThreads = NbThreadsPerWorkGroup;

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckInit, i++, sizeof(cl_mem),  (void *)&d_sync);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckInit, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    {
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum &= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_sync);
        if (__alg)
            ciErrNum &= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckKernel, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return EXIT_SUCCESS;
}

