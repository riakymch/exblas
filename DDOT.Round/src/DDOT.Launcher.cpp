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
#define DDOT_ROUND_KERNEL    "DDOTRound"

static size_t szKernelLength;                  //Byte size of kernel code
static char* cSources = NULL;                  //Buffer to hold source for compilation

static cl_program       cpProgram;             //OpenCL program
static cl_kernel        ckKernel, ckComplete;
static cl_kernel        ckRound;
static cl_command_queue cqDefaultCommandQue;   //Default command queue
static cl_mem           d_Superacc;
static cl_mem           d_PartialSuperaccs;

static const uint PARTIAL_SUPERACCS_COUNT = 1024;
static const uint WORKGROUP_SIZE          = 256;
static const uint MERGE_WORKGROUP_SIZE    = 256;
static const uint VECTOR_NUMBER           = 1;
static uint __alg;

#ifdef AMD
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWORKGROUP_SIZE=256 -DMERGE_WORKGROUP_SIZE=256 -DUSE_KNUTH";
#else
static char  compileOptions[256] = "-DNVIDIA -DWARP_COUNT=16 -DWORKGROUP_SIZE=256 -DMERGE_WORKGROUP_SIZE=256 -DUSE_KNUTH -cl-mad-enable -cl-fast-relaxed-math"; // -cl-nv-verbose";
#endif


extern "C" cl_int initDDOT(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint alg,
    const uint NbFPE
){
    cl_int ciErrNum;
    __alg = alg;

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

    printf("...building Reduction program\n");
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

    printf("...creating Superaccs kernels:\n");
        ckKernel = clCreateKernel(cpProgram, DDOT_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: DDOT, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckComplete = clCreateKernel(cpProgram, DDOT_COMPLETE_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: DDOTComplete, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        if (__alg > 0) {
            ckRound = clCreateKernel(cpProgram, DDOT_ROUND_KERNEL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateKernel: DDOTRound, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }

    printf("...allocating internal buffer\n");
        if (__alg > 0) {
            d_Superacc = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, BIN_COUNT * sizeof(bintype), NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateBuffer for d_Superacc, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
            d_PartialSuperaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, PARTIAL_SUPERACCS_COUNT * BIN_COUNT * sizeof(cl_long), NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        } else { // for reduction
            d_PartialSuperaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, PARTIAL_SUPERACCS_COUNT * sizeof(cl_double), NULL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeDDOT(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseMemObject(d_PartialSuperaccs);
    if (__alg > 0) {
        ciErrNum |= clReleaseMemObject(d_Superacc);
        ciErrNum |= clReleaseKernel(ckRound);
    }
    ciErrNum |= clReleaseKernel(ckKernel);
    ciErrNum |= clReleaseKernel(ckComplete);
    ciErrNum |= clReleaseProgram(cpProgram);

    if (ciErrNum != CL_SUCCESS)
        printf("Error in closeDDOT(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Reduction / mergeReduction kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t DDOT(
    cl_command_queue cqCommandQueue,
    cl_mem d_res,
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
        TotalNbThreads = PARTIAL_SUPERACCS_COUNT * NbThreadsPerWorkGroup;
        NbElements = NbElements / VECTOR_NUMBER;

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint), (void *)&NbElements);
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
        TotalNbThreads = BIN_COUNT * NbThreadsPerWorkGroup;

        uint i = 0;
        if (__alg > 0)
            ciErrNum  = clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_Superacc);
        else
            ciErrNum  = clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_res);
        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_uint), (void *)&PARTIAL_SUPERACCS_COUNT);
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
    if (__alg > 0) {
        NbThreadsPerWorkGroup = MERGE_WORKGROUP_SIZE;
        TotalNbThreads = NbThreadsPerWorkGroup;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(ckRound, i++, sizeof(cl_mem),  (void *)&d_res);
        ciErrNum |= clSetKernelArg(ckRound, i++, sizeof(cl_mem),  (void *)&d_Superacc);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
           *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckRound, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
           *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return WORKGROUP_SIZE;
}

