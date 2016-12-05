/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "common.hpp"
#include "ExGEMV.Launcher.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define DGEMV_KERNEL "dgemv"
#define DGEMVT_KERNEL "dgemvT"
#define GEMV_KERNEL "gemv"
#define GEMVT_KERNEL "gemvT"
#define GEMV_REDUCE "gemv_reduce"

static size_t szKernelLength;                // Byte size of kernel code
static char* cSources = NULL;                // Buffer to hold source for compilation

static cl_program       cpProgram;           //OpenCL program
static cl_kernel        ckGEMV, ckDGEMV;     //OpenCL kernels
static cl_kernel        ckGEMVReduce;        //OpenCL kernels
static cl_command_queue cqDefaultCommandQue; //Default command queue
static cl_mem           d_Superaccs;

static uint p;

static const uint WORKGROUP_SIZE = 256;

#ifdef AMD
static char  compileOptions[256] = "-DUSE_KNUTH -DWORKGROUP_SIZE=256";
#else
static char  compileOptions[256] = "-DNVIDIA -DUSE_KNUTH -DWORKGROUP_SIZE=256 -cl-mad-enable -cl-fast-relaxed-math"; // -cl-nv-verbose";
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExGEMV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
	const char transa,
    const uint m,
    const uint ip,
    const uint NbFPE
){
    cl_int ciErrNum;
    p = ip;

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

    //printf("...building ExGEMV program\n");
        char compileOptionsBak[256];
        sprintf(compileOptionsBak, "%s -DNBFPE=%d", compileOptions, NbFPE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptionsBak, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[8192];
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }

    if (NbFPE == 1){
        ckDGEMV = clCreateKernel(cpProgram, (transa == 'T' ? DGEMVT_KERNEL : DGEMV_KERNEL), &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: dgemv, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
    } else {
        //printf("...creating ExGEMV kernels:\n");
        ckGEMV = clCreateKernel(cpProgram, (transa == 'T' ? GEMVT_KERNEL : GEMV_KERNEL), &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: gemv, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckGEMVReduce = clCreateKernel(cpProgram, GEMV_REDUCE, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: gemv_reduce, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

        //printf("...allocating internal buffer\n");
        d_Superaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, m * p * bin_count * sizeof(cl_long), NULL, &ciErrNum);
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

extern "C" void closeExGEMV(void){
    cl_int ciErrNum = CL_SUCCESS;

    if (d_Superaccs){
        ciErrNum = clReleaseMemObject(d_Superaccs);
        d_Superaccs = NULL;
    }
    if (ckDGEMV) {
        ciErrNum |= clReleaseKernel(ckDGEMV);
        ckDGEMV = NULL;
    }
    if (ckGEMV) {
        ciErrNum |= clReleaseKernel(ckGEMV);
        ckGEMV = NULL;
    }
    if (ckGEMVReduce) {
        ciErrNum |= clReleaseKernel(ckGEMVReduce);
        ckGEMVReduce = NULL;
    }
    ciErrNum |= clReleaseProgram(cpProgram);

    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeExGEMV(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for GEMV kernels
////////////////////////////////////////////////////////////////////////////////
// Non-transpose version
extern "C" size_t ExGEMV(
    cl_command_queue cqCommandQueue,
	const char transa,
    const uint m,
    const uint n,
    const double alpha,
    const cl_mem d_a,
    const uint lda,
    const uint offseta,
    const cl_mem d_x,
    const uint incx,
    const uint offsetx,
    const double beta,
    cl_mem d_y,
    const uint incy,
    const uint offsety,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    if (ckDGEMV) {
        size_t NbThreadsPerWorkGroup[] = {WORKGROUP_SIZE, 1};
        size_t TotalNbThreads[] = {WORKGROUP_SIZE * (((transa == 'T' ? n : m) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE), p};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&m);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_double), (void *)&alpha);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&lda);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&offseta);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&incx);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&offsetx);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_double), (void *)&beta);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_mem),  (void *)&d_y);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&incy);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, sizeof(cl_uint), (void *)&offsety);
        ciErrNum &= clSetKernelArg(ckDGEMV, i++, ((transa == 'T' ? m : n) / p) * sizeof(cl_double), NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckDGEMV, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("ciErrNum = %d\n", ciErrNum);
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
		return 0;
    } else {
        size_t NbThreadsPerWorkGroup[] = {WORKGROUP_SIZE, 1};
        size_t TotalNbThreads[] = {WORKGROUP_SIZE * (((transa == 'T' ? n : m) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE), p};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&m);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&n);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_double), (void *)&alpha);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_a);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&lda);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&offseta);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_x);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&incx);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&offsetx);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_double), (void *)&beta);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_mem),  (void *)&d_y);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&incy);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, sizeof(cl_uint), (void *)&offsety);
        ciErrNum &= clSetKernelArg(ckGEMV, i++, ((transa == 'T' ? m : n) / p) * sizeof(cl_double), NULL);
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
    /*{
        size_t NbThreadsPerWorkGroup[] = {mt, 1};
        size_t TotalNbThreads[] = {m, 1};

        uint i = 0;
        ciErrNum  = clSetKernelArg(ckGEMVReduce, i++, sizeof(cl_uint), (void *)&m);
        ciErrNum &= clSetKernelArg(ckGEMVReduce, i++, sizeof(cl_uint), (void *)&p);
        ciErrNum &= clSetKernelArg(ckGEMVReduce, i++, sizeof(cl_mem),  (void *)&d_Superaccs);
        ciErrNum &= clSetKernelArg(ckGEMVReduce, i++, sizeof(cl_mem),  (void *)&d_y);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckGEMVReduce, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }*/

    return EXIT_SUCCESS;
}

