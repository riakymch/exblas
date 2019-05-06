/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "common.hpp"
#include "ExTRSV.Launcher.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define TRSV_INIT "trsv_init"
#define TRSV_KERNEL "trsv"
#define DTRSV_KERNEL "dtrsv"
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
static cl_kernel        ckDTRSV;             //OpenCL kernels
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
        sprintf(compileOptionsBak, "%s -DNBFPE=%d -DN=%d", compileOptions, NbFPE % 10, n / BLOCK_SIZE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptionsBak, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            size_t log_size;
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *log = (char *) malloc(log_size);
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            printf("%s\n", log);

            return EXIT_FAILURE;
        }

    //printf("...creating ExTRSV kernels:\n");
        ckInit = clCreateKernel(cpProgram, TRSV_INIT, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: trsv_init, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

        if (NbFPE == 1) {
            ckDTRSV = clCreateKernel(cpProgram, DTRSV_KERNEL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
				printf("ciErrNum = %d\n", ciErrNum);
                printf("Error in clCreateKernel: dtrsv, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }
        if (NbFPE >= 20) {
            ckDTRSV = clCreateKernel(cpProgram, DTRSV_KERNEL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateKernel: dtrsv, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }
        if ((NbFPE != 20) && (NbFPE != 1)) {
            ckTRSV = clCreateKernel(cpProgram, TRSV_KERNEL, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateKernel: trsv, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                return EXIT_FAILURE;
            }
        }

        if ((NbFPE >= 10) && (NbFPE <= 28) && (NbFPE != 20)) {
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

    //allocating internal buffer
        if ((NbFPE != 20) && (NbFPE < 30)){
            d_Superaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, n * THREADSY * bin_count * sizeof(cl_long), NULL, &ciErrNum);
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

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeExTRSV(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseKernel(ckInit);
    if (d_Superaccs) {
        ciErrNum |= clReleaseMemObject(d_Superaccs);
        d_Superaccs = NULL;
    }
    if (ckTRSV) {
        ciErrNum |= clReleaseKernel(ckTRSV);
        ckTRSV = NULL;
    }
    if (ckDTRSV) {
        ciErrNum |= clReleaseKernel(ckDTRSV);
        ckDTRSV = NULL;
    }
    if (ckGEMV) {
        ciErrNum |= clReleaseKernel(ckGEMV);
        ckGEMV = NULL;
    }
    if (ckAXPY) {
        ciErrNum |= clReleaseKernel(ckAXPY);
        ckAXPY = NULL;
    }
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
    const uint offseta,
    const cl_mem d_x,
    const uint incx,
    const uint offsetx,
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

    if (ckDTRSV) {
        // DTRSV
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint),  (void *)&n);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint),  (void *)&lda);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint),  (void *)&offseta);
        ciErrNum  = clSetKernelArg(ckDTRSV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint),  (void *)&incx);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint),  (void *)&offsetx);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_mem),  (void *)&d_sync);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint),  NULL);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_double),  NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckDTRSV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    } else {
        // ExTRSV
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint),  (void *)&n);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint),  (void *)&lda);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint),  (void *)&offseta);
        ciErrNum  = clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint),  (void *)&incx);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint),  (void *)&offsetx);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_sync);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_uint),  NULL);
        ciErrNum &= clSetKernelArg(ckTRSV, i++, sizeof(cl_double),  NULL);
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
        size_t NbThreadsPerWorkGroup[] = {256, 1};
        size_t TotalNbThreads[] = {n, 1};

        double alpha = -1.0, beta = 1.0;
        int incx = 1;
        uint i = 0;
        ciErrNum  = clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_double), (void *)&alpha);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&incx);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_double), (void *)&beta);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&incx);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, n * sizeof(cl_double), NULL);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
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
        size_t NbThreadsPerWorkGroup[] = {256, 1};
        size_t TotalNbThreads[] = {n, 1};

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

extern "C" size_t DTRSVExIR(
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
    // DTRSV
    {
        size_t NbThreadsPerWorkGroup[] = {THREADSX, THREADSY};
        size_t TotalNbThreads[] = {n, THREADSY};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckDTRSV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_mem),  (void *)&d_sync);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_int),  NULL);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_double),  NULL);
        ciErrNum &= clSetKernelArg(ckDTRSV, i++, sizeof(cl_uint), (void *)&n);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckDTRSV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    // IR: ExGEMV
    {
        size_t NbThreadsPerWorkGroup[] = {256, 1};
        size_t TotalNbThreads[] = {n, 1};

        double alpha = -1.0, beta = 1.0;
        int incx = 1;
        uint i = 0;
        ciErrNum  = clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_double), (void *)&alpha);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&incx);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_double), (void *)&beta);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_b);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&incx);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, n * sizeof(cl_double), NULL);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
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
        size_t NbThreadsPerWorkGroup[] = {256, 1};
        size_t TotalNbThreads[] = {n, 1};

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

