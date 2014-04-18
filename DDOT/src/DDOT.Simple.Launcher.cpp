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
#define DDOT_KERNEL          "DDOT"
#define DDOT_COMPLETE_KERNEL "DDOTComplete"

static size_t szKernelLength;		       // Byte size of kernel code
static char* cSources = NULL;                  // Buffer to hold source for compilation

static cl_program       cpProgram;             //OpenCL Reduction program
static cl_kernel        ckKernel, ckComplete;
static cl_command_queue cqDefaultCommandQue;   //Default command queue for Reduction
static cl_mem           d_PartialSum;

static const uint  PARTIAL_SUMMATION_COUNT = 2048;
static const uint  WARP_SIZE               = 32;
static const uint  WORKGROUP_SIZE          = 256;
static const uint  MERGE_WORKGROUP_SIZE    = 256;
static const uint  VECTOR_NUMBER           = 1;

#ifdef AMD
static char  compileOptions[256] = "-DWARP_SIZE=16 -DWORKGROUP_SIZE=256 -DMERGE_WORKGROUP_SIZE=256";
#else
static char  compileOptions[256] = "-DWARP_SIZE=16 -DWORKGROUP_SIZE=256 -DMERGE_WORKGROUP_SIZE=256 -cl-mad-enable -cl-fast-relaxed-math";
//static char  compileOptions[256] = "-DWARP_SIZE=16 -DWORKGROUP_SIZE=256 -DMERGE_WORKGROUP_SIZE=256 -cl-nv-verbose";
#endif


extern "C" cl_int initDDOTSimple(
    cl_context cxGPUContext, 
    cl_command_queue cqParamCommandQue, 
    cl_device_id cdDevice,
    const char* program_file
){
    cl_int ciErrNum;
    size_t kernelLength;

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
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSources, &kernelLength, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    printf("...building Reduction program\n");
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            //printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[4096];
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }

    printf("...creating Reduction kernels:\n");
        ckKernel = clCreateKernel(cpProgram, DDOT_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: Reduction, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckComplete = clCreateKernel(cpProgram, DDOT_COMPLETE_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: Reduction_Complete, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
    printf("...allocating internal buffer\n");
        d_PartialSum = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, PARTIAL_SUMMATION_COUNT * sizeof(cl_double), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeDDOTSimple(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseMemObject(d_PartialSum);
    ciErrNum |= clReleaseKernel(ckKernel);
    ciErrNum |= clReleaseKernel(ckComplete);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeReduction(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Reduction / mergeReduction kernels
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b){
    return a - a % b;
}

extern "C" size_t DDOTSimple(
    cl_command_queue cqCommandQueue,
    cl_mem d_FinalSum,
    const cl_mem d_a,
    const cl_mem d_b,
    uint NbElements,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;
    size_t NbThreadsPerWorkGroup, TotalNbThreads;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        NbThreadsPerWorkGroup  = WORKGROUP_SIZE;
        TotalNbThreads = PARTIAL_SUMMATION_COUNT * NbThreadsPerWorkGroup;
        NbElements = NbElements / VECTOR_NUMBER;

        ciErrNum  = clSetKernelArg(ckKernel, 0, sizeof(cl_mem),  (void *)&d_PartialSum);
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_uint), (void *)&NbElements);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    {
        NbThreadsPerWorkGroup = MERGE_WORKGROUP_SIZE;
        TotalNbThreads = NbThreadsPerWorkGroup;

        ciErrNum  = clSetKernelArg(ckComplete, 0, sizeof(cl_mem),  (void *)&d_FinalSum);
        ciErrNum |= clSetKernelArg(ckComplete, 1, sizeof(cl_mem),  (void *)&d_PartialSum);
        ciErrNum |= clSetKernelArg(ckComplete, 2, sizeof(cl_uint), (void *)&PARTIAL_SUMMATION_COUNT);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckComplete, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return WORKGROUP_SIZE;
}
