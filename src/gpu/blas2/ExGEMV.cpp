/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file gpu/blas2/ExGEMV.cpp
 *  \brief Provides implementations of a set of gemv routines
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
#include "ExGEMV.Launcher.hpp"


#define NUM_ITER 5


/**
 * \ingroup ExGEMV
 * \brief Executes on GPU parallel matrix-vector operations y := alpha*A*x + beta * y or y := alpha*A**T*x + beta * y.
 *     For internal use
 *
 * \param transa 'T' or 'N' a transpose or a non-transpose matrix A
 * \param m the number of rows of matrix A
 * \param n the number of columns of matrix A
 * \param alpha scalar
 * \param a matrix A
 * \param lda leading dimension of A
 * \param offseta specifies position in the metrix A from its beginning
 * \param x vector
 * \param incx the increment for the elements of a
 * \param offsetx specifies position in the vector x from its start
 * \param beta scalar
 * \param y vector
 * \param incy the increment for the elements of a
 * \param offsety specifies position in the vector y from its start
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return Contains the reproducible and accurate result of the matrix-vector product
 */
int runExGEMV(const char transa, const int m, const int n, const double alpha, double *a, const int lda, const int offseta, double *x, const int incx, const int offsetx, const double beta, double *y, const int incy, const int offsety, const int fpe, const char* program_file);

/**
 * \ingroup ExGEMV
 * \brief Parallel gemv based on our algorithm. If fpe < 3, use superaccumulators only.
 *  Otherwise, use floating-point expansions of size FPE with superaccumulators when needed.
 *  early_exit corresponds to the early-exit technique. For now, it works with non-transpose matrices
 *
 * \param transa 'T' or 'N' a transpose or a non-transpose matrix A
 * \param m the number of rows of matrix A
 * \param n the number of columns of matrix A
 * \param alpha scalar
 * \param a matrix A
 * \param lda leading dimension of A
 * \param offseta specifies position in the metrix A from its beginning
 * \param x vector
 * \param incx the increment for the elements of a
 * \param offsetx specifies position in the vector x from its start
 * \param beta scalar
 * \param y vector
 * \param incy the increment for the elements of a
 * \param offsety specifies position in the vector y from its start
 * \param fpe size of floating-point expansion
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return matrix C contains the reproducible and accurate result of the matrix product
 */
int exgemv(const char transa, const int m, const int n, const double alpha, double *a, const int lda, const int offseta, double *x, const int incx, const int offsetx, const double beta, double *y, const int incy, const int offsety, const int fpe, const bool early_exit) {
    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // with superaccumulators only
    if (fpe == 0) {
        return runExGEMV(transa, m, n, alpha,  a, lda, offseta, x, incx, offsetx, beta, y, incy, offsety, 0, strcat(path, "ExGEMV.Superacc.cl"));
    }

    // DGEMV
    if (fpe == 1) {
        return runExGEMV(transa, m, n, alpha,  a, lda, offseta, x, incx, offsetx, beta, y, incy, offsety, 1, strcat(path, "DGEMV.cl"));
    }

    if (early_exit) {
        if (fpe <= 4)
            return runExGEMV(transa, m, n, alpha, a, lda, offseta, x, incx, offsetx, beta, y, incy, offsety, 4, strcat(path, "ExGEMV.FPE.EX.4.cl"));
        if (fpe <= 6)
            return runExGEMV(transa, m, n, alpha, a, lda, offseta, x, incx, offsetx, beta, y, incy, offsety,6, strcat(path, "ExGEMV.FPE.EX.6.cl"));
        if (fpe <= 8)
            return runExGEMV(transa, m, n, alpha, a, lda, offseta, x, incx, offsetx, beta, y, incy, offsety, 8, strcat(path, "ExGEMV.FPE.EX.8.cl"));
    } else // ! early_exit
        return runExGEMV(transa, m, n, alpha, a, lda, offseta, x, incx, offsetx, beta, y, incy, offsety, fpe, strcat(path, "ExGEMV.FPE.cl"));

    return 0;
}

int runExGEMV(const char transa, const int m, const int n, const double alpha, double *a, const int lda, const int offseta, double *x, const int incx, const int offsetx, const double beta, double *y, const int incy, const int offsety, const int fpe, const char* program_file) {
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
        cl_mem d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * n * sizeof(cl_double), a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_x = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ((transa == 'T') ? m : n) * sizeof(cl_double), x, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_x, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_y = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ((transa == 'T') ? n : m) * sizeof(cl_double), y, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_y, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

    {
        //Initializing OpenCL dSum...
        ciErrNum = initExGEMV(cxGPUContext, cqCommandQueue, cdDevice, program_file, transa, (transa == 'T') ? n : m, 1, fpe);
        if (ciErrNum != CL_SUCCESS)
            exit(EXIT_FAILURE);

        //Running OpenCL exgemv with %u elements...
        ExGEMV(NULL, transa, m, n, alpha, d_a, lda, offseta, d_x, incx, offsetx, beta, d_y, incy, offsety, &ciErrNum);
        if (ciErrNum != CL_SUCCESS)
            exit(EXIT_FAILURE);

#ifdef EXBLAS_TIMING
        double t, mint = 10000;
        cl_event startMark, endMark;

        for(uint iter = 0; iter < NUM_ITER; iter++) {
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
            //ciErrNum = clEnqueueMarkerWithWaitList(cqCommandQueue, 0, NULL, &startMark);
            ciErrNum |= clFinish(cqCommandQueue);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueMarker, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

            ExGEMV(NULL, transa, m, n, alpha, d_a, lda, offseta, d_x, incx, offsetx, beta, d_y, incy, offsety, &ciErrNum);

            ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
            //ciErrNum = clEnqueueMarkerWithWaitList(cqCommandQueue, 0, NULL, &endMark);
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
            t = 1e-9 * ((unsigned long)endTime - (unsigned long)startTime);
            mint = std::min(t, mint);
        }

        double perf = 2 * m * n;
        perf = (perf / mint) * 1e-9;
        double throughput = (m * n + m + n) * sizeof(double);
        throughput = (throughput / mint) * 1e-9;
        printf("NbFPE = %u \t M = %u \t N = %u \t Time = %.8f s \t Throughput = %.4f GB/s \t Performance = %.4f GFLOPS\n", fpe, m, n, mint, throughput, perf);
        fprintf(stderr, "%f  ", mint);
#endif

        //Retrieving results...
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_y, CL_TRUE, 0, ((transa == 'T') ? n : m) * sizeof(cl_double), y, 0, NULL, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("ciErrNum = %d\n", ciErrNum);
                printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

         //Release kernels and program
         //Shutting down and freeing memory...
            closeExGEMV();
            clReleaseMemObject(d_a);
            clReleaseMemObject(d_x);
            clReleaseMemObject(d_y);
            clReleaseCommandQueue(cqCommandQueue);
            clReleaseContext(cxGPUContext);
    }

    return EXIT_SUCCESS;
}

