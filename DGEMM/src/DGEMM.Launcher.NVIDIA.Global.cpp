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
#define DGEMM_KERNEL "matrixMul"
#ifdef AMD
  #define BLOCK_SIZE 16
#else
  #define BLOCK_SIZE 32
#endif

static size_t szKernelLength;	              // Byte size of kernel code
static char* cSources = NULL;                 // Buffer to hold source for compilation

static cl_program       cpProgram;            //OpenCL Superaccumulator program
static cl_kernel        ckMatrixMul;
static cl_command_queue cqDefaultCommandQue;  //Default command queue for Superaccumulator
static cl_mem 		d_Accus;

static const uint  VECTOR_NUMBER = 1;

#ifdef AMD
static char  compileOptions[256] = "-DBLOCK_SIZE=16";
#else
static char  compileOptions[256] = "-DBLOCK_SIZE=32 -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math -cl-nv-verbose";
#endif


extern "C" cl_int initDGEMMNVIDIAGlobal(
    cl_context cxGPUContext, 
    cl_command_queue cqParamCommandQue, 
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE,
    const uint width,
    const uint height
){
    cl_int ciErrNum;
    size_t kernelLength;

    // Read the OpenCL kernel from source file
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

    printf("...building program\n");
	sprintf(compileOptions, "%s -DNBFPE=%d", compileOptions, NbFPE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        //if (ciErrNum != CL_SUCCESS) {
            //printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[4096]; // 16384
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

        //    return EXIT_FAILURE;
        //}

    printf("...creating DGEMM kernel:\n");
        ckMatrixMul = clCreateKernel(cpProgram, DGEMM_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    printf("...allocating memory for a matrix of superaccumualtors\n");
        size_t size = width * height * BIN_COUNT * sizeof(bintype);
        d_Accus = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_C, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeDGEMMNVIDIAGlobal(void){
    cl_int ciErrNum;

    if(d_Accus) 
	clReleaseMemObject(d_Accus);

    ciErrNum = clReleaseKernel(ckMatrixMul);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeDGEMM(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Superaccumulator / mergeSuperaccumulators kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t DGEMMNVIDIAGlobal(
    cl_command_queue cqCommandQueue,
    Matrix d_C,
    const Matrix d_A,
    const Matrix d_B,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        //size_t NbThreadsPerWorkGroup[] = {BLOCK_SIZE, BLOCK_SIZE / VECTOR_NUMBER};
        size_t NbThreadsPerWorkGroup[] = {BLOCK_SIZE, BLOCK_SIZE};
	size_t widthC = d_C.width / VECTOR_NUMBER;
	size_t heightC = d_C.height / VECTOR_NUMBER;
	size_t TotalNbThreads[] = {widthC, heightC};
	size_t neededLocalMemory = (BLOCK_SIZE * VECTOR_NUMBER) * (BLOCK_SIZE * VECTOR_NUMBER) * sizeof(cl_double);

	cl_int i = 0;
        ciErrNum  = clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_Accus);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_C.elements);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_A.elements);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_B.elements);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_int),  (void *)&d_A.width);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_int),  (void *)&d_B.width);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, neededLocalMemory,  NULL);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, neededLocalMemory,  NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckMatrixMul, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, 0, 0);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    return EXIT_SUCCESS;
}

