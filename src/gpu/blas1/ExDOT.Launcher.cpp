/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "common.hpp"
#include "ExDOT.Launcher.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define EXDOT_KERNEL          "ExDOT"
#define EXDOT_COMPLETE_KERNEL "ExDOTComplete"

static size_t szKernelLength;                // Byte size of kernel code
static char* cSources = NULL;                // Buffer to hold source for compilation

static cl_program       cpProgram;           //OpenCL Superaccumulator program
static cl_kernel        ckKernel;            //OpenCL Superaccumulator kernels
static cl_kernel        ckComplete;
static cl_command_queue cqDefaultCommandQue; //Default command queue for Superaccumulator
static cl_mem           d_PartialSuperaccs;

#ifdef AMD
static const uint PARTIAL_SUPERACCS_COUNT = 768;
#else
static const uint PARTIAL_SUPERACCS_COUNT = 512;
#endif
static const uint WORKGROUP_SIZE          = 256;
static const uint MERGE_WORKGROUP_SIZE    = 64;
static const uint MERGE_SUPERACCS_SIZE    = 128;

#ifdef AMD
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=64 -DMERGE_SUPERACCS_SIZE=128 -DUSE_KNUTH";
#else
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=64 -DMERGE_SUPERACCS_SIZE=128 -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExDOT(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE
){
    cl_int ciErrNum;

    // Read the OpenCL kernel in from source file
    FILE *program_handle;
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

    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSources, &szKernelLength, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return EXIT_FAILURE;
    }

    sprintf(compileOptions, "%s -DNBFPE=%d", compileOptions, NbFPE);
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
        printf("%s\n", buildLog);

        return EXIT_FAILURE;
    }

    ckKernel = clCreateKernel(cpProgram, EXDOT_KERNEL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in clCreateKernel: ExDOT, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return EXIT_FAILURE;
    }
    ckComplete = clCreateKernel(cpProgram, EXDOT_COMPLETE_KERNEL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in clCreateKernel: ExDOTComplete, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return EXIT_FAILURE;
    }

    uint size = PARTIAL_SUPERACCS_COUNT;
    size = size * bin_count * sizeof(cl_long);
    d_PartialSuperaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in clCreateBuffer for d_PartialSuperaccs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        return EXIT_FAILURE;
    }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeExDOT(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseMemObject(d_PartialSuperaccs);
    ciErrNum |= clReleaseKernel(ckKernel);
    ciErrNum |= clReleaseKernel(ckComplete);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeExDOT(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers
////////////////////////////////////////////////////////////////////////////////
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
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;
    size_t NbThreadsPerWorkGroup, TotalNbThreads;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        NbThreadsPerWorkGroup  = WORKGROUP_SIZE;
        TotalNbThreads = PARTIAL_SUPERACCS_COUNT * NbThreadsPerWorkGroup;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&inca);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&offseta);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&incb);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&offsetb);
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
        TotalNbThreads = NbThreadsPerWorkGroup;
        TotalNbThreads *= PARTIAL_SUPERACCS_COUNT / MERGE_SUPERACCS_SIZE;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_Res);
        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_uint), (void *)&PARTIAL_SUPERACCS_COUNT);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckComplete, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return WORKGROUP_SIZE;
}


extern double ExDOT(
    cl_uint NbElements,
    cl_mem d_a,
    const cl_uint inca,
    const cl_uint offseta,
    cl_mem d_b,
    const cl_uint incb,
    const cl_uint offsetb,
    cl_command_queue cqCommandQueue
){
    cl_int ciErrNum;
    size_t NbThreadsPerWorkGroup, TotalNbThreads;
    double h_Res = 0.0;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        NbThreadsPerWorkGroup  = WORKGROUP_SIZE;
        TotalNbThreads = PARTIAL_SUPERACCS_COUNT * NbThreadsPerWorkGroup;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&inca);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&offseta);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&incb);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint),  (void *)&offsetb);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint), (void *)&NbElements);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
    }

    {
        NbThreadsPerWorkGroup = MERGE_WORKGROUP_SIZE;
        TotalNbThreads = NbThreadsPerWorkGroup;
        TotalNbThreads *= PARTIAL_SUPERACCS_COUNT / MERGE_SUPERACCS_SIZE;

//        cl_mem d_Res = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ciErrNum);
//        if (ciErrNum != CL_SUCCESS) {
//            printf("Error in clCreateBuffer for d_res, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
//            exit(EXIT_FAILURE);
//        }
//
//        cl_uint i = 0;
//        ciErrNum  = clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_Res);
//        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
//        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_uint), (void *)&PARTIAL_SUPERACCS_COUNT);
//        if (ciErrNum != CL_SUCCESS) {
//            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
//            exit(EXIT_FAILURE);
//        }
//
//        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckComplete, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
//        if (ciErrNum != CL_SUCCESS) {
//            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
//            exit(EXIT_FAILURE);
//        }
//
//        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Res, CL_TRUE, 0, sizeof(cl_double), &h_Res, 0, NULL, NULL);
//        if (ciErrNum != CL_SUCCESS) {
//            printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
//            exit(EXIT_FAILURE);
//        }
    }

    return h_Res;
}
