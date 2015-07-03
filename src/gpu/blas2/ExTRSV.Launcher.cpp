/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "common.hpp"
#include "ExTRSV.Launcher.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define TRSV_INIT "trsv_init"
#define TRSV_KERNEL "trsv"
#define GEMV_KERNEL "gemv"
#define AXPY_KERNEL "axpy"
#ifdef AMD
  #define THREADSX   32
  #define THREADSY    1
  #define BLOCK_SIZE 32
#else
  #define THREADSX   32
  #define THREADSY    1
  #define BLOCK_SIZE 32
#endif

static size_t szKernelLength;                // Byte size of kernel code
static char* cSources = NULL;                // Buffer to hold source for compilation

static cl_program       cpProgram;           //OpenCL program
static cl_kernel        ckInit, ckTRSV;      //OpenCL kernels
static cl_kernel        ckAXPY, ckGEMV;      //OpenCL kernels
static cl_command_queue cqDefaultCommandQue; //Default command queue
static cl_mem           d_sync;
static cl_mem           d_Superaccs;

#ifdef AMD
static char  compileOptions[256] = "-DUSE_KNUTH -DBLOCK_SIZE=32 -Dthreadsx=32 -Dthreadsy=1";
#else
//static char  compileOptions[256] = "-DNVIDIA -DUSE_KNUTH -DBLOCK_SIZE=32 -Dthreadsx=32 -Dthreadsy=1 -cl-mad-enable -cl-fast-relaxed-math"; // -cl-nv-verbose";
static char  compileOptions[256] = "-DNVIDIA -DUSE_KNUTH -DBLOCK_SIZE=32 -Dthreadsx=32 -Dthreadsy=1 -cl-mad-enable";
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExTRSV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint n,
    const uint NbFPE
){
    cl_int ciErrNum;

    // Read the OpenCL kernel in from source file
    FILE *program_handle;
    //printf("Load the program sources (%s)...\n", program_file);
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

    //printf("...building ExTRSV program\n");
        char compileOptionsBak[256];
        sprintf(compileOptionsBak, "%s -DNBFPE=%d -DN=%d", compileOptions, NbFPE, n / BLOCK_SIZE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptionsBak, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[8192];
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }

    //printf("...creating ExTRSV kernels:\n");
        ckInit = clCreateKernel(cpProgram, TRSV_INIT, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: trsv_init, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckTRSV = clCreateKernel(cpProgram, TRSV_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: trsv_lnn, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        if (NbFPE == 1) {
            ckGEMV = clCreateKernel(cpProgram, GEMV_KERNEL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateKernel: gemv, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
            ckAXPY = clCreateKernel(cpProgram, AXPY_KERNEL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateKernel: axpy, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }

    //printf("...allocating internal buffer\n");
        d_Superaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, n * THREADSY * bin_count * sizeof(cl_long), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        d_sync = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, 2 * sizeof(cl_int), NULL, &ciErrNum);
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

extern "C" void closeExTRSV(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseMemObject(d_Superaccs);
    ciErrNum |= clReleaseKernel(ckInit);
    ciErrNum |= clReleaseKernel(ckTRSV);
    if (ckGEMV)
        ciErrNum |= clReleaseKernel(ckGEMV);
    if (ckAXPY)
        ciErrNum |= clReleaseKernel(ckAXPY);
    ciErrNum |= clReleaseProgram(cpProgram);

    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeExTRSV(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for TRSV kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t ExTRSV(
    cl_command_queue cqCommandQueue,
    const uint n,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_x,
    const uint incx,
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
        //size_t NbThreadsPerWorkGroup[] = {1, 1};
        //size_t TotalNbThreads[] = {1, 1};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_sync);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_int),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTRSV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return EXIT_SUCCESS;
}

extern "C" size_t ExTRSVIR(
    cl_command_queue cqCommandQueue,
    const uint n,
    const cl_mem d_a,
    const uint lda,
    const cl_mem d_x,
    const uint incx,
    const cl_mem d_b,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;
    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    // TRSV init
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
    // ExTRSV
    {
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_sync);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_int),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTRSV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    // IR: ExGEMV
    {
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckGEMV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    // IR: ExTRSV -- init
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
    // IR: ExTRSV
    {
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_sync);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_int),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckTRSV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    // IR: ExAXPY
    {
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckAXPY, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckAXPY, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum &= clSetKernelArg(ckAXPY, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckAXPY, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return EXIT_SUCCESS;
}

