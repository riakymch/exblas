/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file gpu/blas2/ExTRSV.cpp
 *  \brief Provides implementations of a set of trsv routines
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>

#include "config.h"
#include "common.hpp"
#include "common.gpu.hpp"
#include "blas2.hpp"
#include "ExTRSV.Launcher.hpp"

#ifdef EXBLAS_TIMING
#include <cassert>

#define NUM_ITER 20

static double min(double arr[], int size) {
    assert(arr != NULL);
    assert(size >= 0);

    if ((arr == NULL) || (size <= 0))
       return NAN;

    double val = DBL_MAX; 
    for (int i = 0; i < size; i++)
        if (val > arr[i])
            val = arr[i];

    return val;
}
#endif

/**
 * \ingroup ExTRSV
 * \brief Executes on GPU parallel triangular solver A*x = b or A**T*x = b.
 *     For internal use
 *
 * \param n size of matrix A
 * \param a matrix A
 * \param lda leading dimension of A
 * \param x vector
 * \param incx the increment for the elements of a
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return Contains the reproducible and accurate result of solving triangular system
 */
int runExTRSV(int n, double *a, int lda, double *x, int incx, int fpe, const char* program_file);

/**
 * \ingroup ExTRSV
 * \brief Parallel TRSV based on our algorithm. If fpe < 3, use superaccumulators only.
 *  Otherwise, use floating-point expansions of size FPE with superaccumulators when needed.
 *  early_exit corresponds to the early-exit technique. For now, it works with non-transpose matrices
 *
 * \param uplo 'U' or 'L' an upper or a lower triangular matrix A
 * \param transa 'T' or 'N' a transpose or a non-transpose matrix A
 * \param diag 'U' or 'N' a unit or non-unit triangular matrix A
 * \param n size of matrix A
 * \param a matrix A
 * \param lda leading dimension of A
 * \param x vector
 * \param incx the increment for the elements of a
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
int extrsv(char uplo, char transa, char diag, int n, double *a, int lda, double *x, int incx, int fpe, bool early_exit) {
    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // with superaccumulators only
    if (fpe == 0)
        return runExTRSV(n, a, lda, x, incx, 0, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.Superacc.cl") : strcat(path, "ExTRSV.unn.Superacc.cl"));

    if (early_exit) {
        if (fpe <= 4)
            //return runExTRSV(n, a, lda, x, incx, 4, strcat(path, "ExTRSV.FPE.EX.4.cl"));
            return runExTRSV(n, a, lda, x, incx, 4, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.4.cl") : strcat(path, "ExTRSV.unn.FPE.EX.4.cl"));
        if (fpe <= 6)
            //return runExTRSV(n, a, lda, x, incx, 6, strcat(path, "ExTRSV.FPE.EX.6.cl"));
            return runExTRSV(n, a, lda, x, incx, 6, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.6.cl") : strcat(path, "ExTRSV.unn.FPE.EX.6.cl"));
        if (fpe <= 8)
            //return runExTRSV(n, a, lda, x, incx, 8, strcat(path, "ExTRSV.FPE.EX.8.cl"));
            return runExTRSV(n, a, lda, x, incx, 8, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.8.cl") : strcat(path, "ExTRSV.unn.FPE.EX.8.cl"));
    } else // ! early_exit
        //return runExTRSV(n, a, lda, x, incx, fpe, strcat(path, "ExTRSV.FPE.cl"));
        return runExTRSV(n, a, lda, x, incx, fpe, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.cl") : strcat(path, "ExTRSV.unn.FPE.cl"));

    return 0;
}

int runExTRSV(int n, double *a, int lda, double *x, int incx, int fpe, const char* program_file){
    cl_int ciErrNum;

    //printf("Initializing OpenCL...\n");
        char platform_name[64];
#ifdef AMD
        strcpy(platform_name, "AMD Accelerated Parallel Processing");
#else
        strcpy(platform_name, "NVIDIA CUDA");
#endif
        cl_platform_id cpPlatform = GetOCLPlatform(platform_name);
        if (cpPlatform == NULL) {
            printf("ERROR: Failed to find the platform '%s' ...\n", platform_name);
            return -1;
        }

        //Get a GPU device
        cl_device_id cdDevice = GetOCLDevice(cpPlatform);
        if (cdDevice == NULL) {
            printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return -1;
        }

        //Create the context
        cl_context cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

        //Create a command-queue
        cl_command_queue cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

        //Allocating OpenCL memory...
        cl_mem d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * n * sizeof(cl_double), a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_x = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * sizeof(cl_double), x, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_x, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_b;
        if (fpe == 1) {
            // for IR case
            d_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, n * sizeof(cl_double), x, &ciErrNum);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clCreateBuffer for d_b, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }
        }


    {
        //Initializing OpenCL dSum...
        ciErrNum = initExTRSV(cxGPUContext, cqCommandQueue, cdDevice, program_file, n, fpe);
        if (ciErrNum != CL_SUCCESS)
            exit(EXIT_FAILURE);

        //Running OpenCL dSum with %u elements...
        if (fpe == 1) {
            // for IR case
            ExTRSVIR(NULL, n, d_a, lda, d_x, incx, d_b, &ciErrNum);
        } else {
            ExTRSV(NULL, n, d_a, lda, d_x, incx, &ciErrNum);
        }
        if (ciErrNum != CL_SUCCESS)
            exit(EXIT_FAILURE);

#ifdef EXBLAS_TIMING
        double gpuTime[NUM_ITER];
        cl_event startMark, endMark;

        for(uint iter = 0; iter < NUM_ITER; iter++) {
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
            ciErrNum |= clFinish(cqCommandQueue);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueMarker, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

            if (fpe == 1) {
                // for IR case
                ExTRSVIR(NULL, n, d_a, lda, d_x, incx, d_b, &ciErrNum);
            } else {
                ExTRSV(NULL, n, d_a, lda, d_x, incx, &ciErrNum);
            }

            ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
            ciErrNum |= clFinish(cqCommandQueue);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueMarker, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }
            //Get OpenCL profiler time
            cl_ulong startTime = 0, endTime = 0;
            ciErrNum  = clGetEventProfilingInfo(startMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &startTime, NULL);
            ciErrNum |= clGetEventProfilingInfo(endMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clGetEventProfilingInfo Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }
            gpuTime[iter] = 1e-9 * ((unsigned long)endTime - (unsigned long)startTime); // / (double)NUM_ITER;
        }

        double minTime = min(gpuTime, NUM_ITER);
        double perf = n * n;
        perf = (perf / minTime) * 1e-9;
        printf("NbFPE = %u \t N = %u \t \t Time = %.8f s \t Performance = %.4f GFLOPS\n", fpe, n, minTime, perf);
#endif

        //Retrieving results...
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_x, CL_TRUE, 0, n * sizeof(cl_double), x, 0, NULL, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("ciErrNum = %d\n", ciErrNum);
                printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

         //Release kernels and program
         //Shutting down and freeing memory...
            closeExTRSV();
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_x);
            if(fpe == 1)
                clReleaseMemObject(d_b);
            clReleaseCommandQueue(cqCommandQueue);
            clReleaseContext(cxGPUContext);
    }

    return EXIT_SUCCESS;
}

