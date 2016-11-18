/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "common.hpp"
#include "ExGEMM.Launcher.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define DGEMM_KERNEL "gemm"
#ifdef AMD
  #define BLOCK_SIZE 16
#else
  #define BLOCK_SIZE 32
#endif

static size_t szKernelLength;                 // Byte size of kernel code
static char* cSources = NULL;                 // Buffer to hold source for compilation

static cl_program       cpProgram;            //OpenCL Superaccumulator program
static cl_kernel        ckMatrixMul;
static cl_command_queue cqDefaultCommandQue;  //Default command queue for Superaccumulator

#ifdef AMD
static char  compileOptions[256] = "-DBLOCK_SIZE=16 -DUSE_KNUTH";
#else
static char  compileOptions[256] = "-DBLOCK_SIZE=32 -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExGEMM(
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

    //printf("clCreateProgramWithSource...\n"); 
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSources, &szKernelLength, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //printf("...building program\n");
        sprintf(compileOptions, "%s -DNBFPE=%d", compileOptions, NbFPE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[4096]; // 16384
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }

    //printf("...creating DGEMM kernel:\n");
        ckMatrixMul = clCreateKernel(cpProgram, DGEMM_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeExGEMM(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseKernel(ckMatrixMul);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeDGEMM(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for the kernel
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t ExGEMM(
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
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        // this parameter tells how we divide the matrix to perform computations
        double multi = 1.0;
        size_t NbThreadsPerWorkGroup[] = {(size_t) BLOCK_SIZE, (size_t) BLOCK_SIZE};
        size_t TotalNbThreads[] = {(size_t) (m), (size_t) (n / multi)};
        size_t neededLocalMemory = BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double);

        cl_int i = 0;
        ciErrNum  = clSetKernelArg(ckMatrixMul, i++, sizeof(cl_uint), (void *)&m);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_uint), (void *)&k);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_double), (void *)&alpha);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem), (void *)&d_a);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_uint), (void *)&lda);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem), (void *)&d_b);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_uint), (void *)&ldb);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_double), (void *)&beta);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem), (void *)&d_c);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_uint), (void *)&ldc);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, neededLocalMemory,  NULL);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, neededLocalMemory,  NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckMatrixMul, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, 0, 0);
        if (ciErrNum != CL_SUCCESS) {
            ciErrNum == -5 ? printf("\nThere is a failure to allocate resources required by the OpenCL implementation on the device\n\n") : printf("ciErrNum = %d\n", ciErrNum);
            //printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return EXIT_SUCCESS;
}

