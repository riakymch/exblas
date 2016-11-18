/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file gpu/blas3/ExGEMM.cpp
 *  \brief Provides implementations of a set of gemm routines
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
#include "blas3.hpp"
#include "ExGEMM.Launcher.hpp"


#define NUM_ITER 20


/**
 * \ingroup ExGEMM
 * \brief Executes on GPU parallel matrix-matrix multiplication (C := beta * C + alpha * op(A) * op(B), where op(X) = X or op(X) = X^T).
 *  For internal use
 *
 * \param m nb of rows of matrix C
 * \param n nb of columns of matrix C
 * \param k nb of rows in matrix B
 * \param alpha scalar
 * \param a matrix A
 * \param lda leading dimension of A
 * \param b matrix B
 * \param ldb leading dimension of B
 * \param beta scalar
 * \param c matrix C
 * \param ldc leading dimension of C
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return matrix C contains the reproducible and accurate result of the matrix product
 */
static int runExGEMM(int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc, int fpe, const char* program_file);


/**
 * \ingroup ExGEMM
 * \brief Parallel GEMM based on our algorithm. If fpe < 3, use superaccumulators only.
 *  Otherwise, use floating-point expansions of size FPE with superaccumulators when needed.
 *  early_exit corresponds to the early-exit technique. For now, it works on non-transpose matrices
 *
 * \param transa 'T' or 'N' a transpose or a non-transpose matrix A
 * \param transb 'T' or 'N' a transpose or a non-transpose matrix B
 * \param m nb of rows of matrix C
 * \param n nb of columns of matrix C
 * \param k nb of rows in matrix B
 * \param alpha scalar
 * \param a matrix A
 * \param lda leading dimension of A
 * \param b matrix B
 * \param ldb leading dimension of B
 * \param beta scalar
 * \param c matrix C
 * \param ldc leading dimension of C
 * \param fpe size of FPE
 * \param early_exit Flag to indicate the early-exit technique
 * \return status
 */
int exgemm(char transa, char transb, int m, int n, int k, double alpha, double *a, int lda, double *b, int ldb, double beta, double *c, int ldc, int fpe, bool early_exit) {
    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // with superaccumulators only
    if (fpe < 3) {
        return runExGEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 0, strcat(path, "ExGEMM.Superacc.cl"));
    }

    if (early_exit) {
        if (fpe <= 4)
            return runExGEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 4, strcat(path, "ExGEMM.FPE.EX.4.cl"));
        if (fpe <= 6)
            return runExGEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 6, strcat(path, "ExGEMM.FPE.EX.6.cl"));
        if (fpe <= 8)
            return runExGEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, 8, strcat(path, "ExGEMM.FPE.EX.8.cl"));
    } else // ! early_exit
        return runExGEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, fpe, strcat(path, "ExGEMM.FPE.cl"));

    return EXIT_SUCCESS;
}

static int runExGEMM(int m, int n, int k, double alpha, double *h_a, int lda, double *h_b, int ldb, double beta, double *h_c, int ldc, int fpe, const char* program_file) {
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
        cl_mem d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m * k * sizeof(cl_double), h_a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, k * n * sizeof(cl_double), h_b, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_b, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_c = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, m * n * sizeof(cl_double), h_c, &ciErrNum);
        //cl_mem d_c = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, m * n * sizeof(cl_double), h_c, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_c, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

    {
        //Initializing OpenCL ExGEMM...
            ciErrNum = initExGEMM(cxGPUContext, cqCommandQueue, cdDevice, program_file, fpe);
            if (ciErrNum != CL_SUCCESS)
                exit(EXIT_FAILURE);

        //Running OpenCL ExGEMM...
            //Just a single launch or a warmup iteration
            ExGEMM(NULL, m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, &ciErrNum);
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

            ExGEMM(NULL, m, n, k, alpha, d_a, lda, d_b, ldb, beta, d_c, ldc, &ciErrNum);

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

        double perf = 2.0 * m * n * k;
        perf = (perf / mint) * 1e-9;
        printf("NbFPE = %u \t M = %u \t N = %u \t K = %u \t Time = %.8f s \t Performance = %.4f GFLOPS\n", fpe, m, n, k, mint, perf);
	fprintf(stderr, "%f ", mint);
#endif

        //Retrieving results...
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_c, CL_TRUE, 0, m * n * sizeof(double), h_c, 0, NULL, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

         //Release kernels and program
         //Shutting down and freeing memory...
            closeExGEMM();
            if(d_a)
                clReleaseMemObject(d_a);
            if(d_b)
                clReleaseMemObject(d_b);
            if(d_c)
                clReleaseMemObject(d_c);
            if(cqCommandQueue)
                clReleaseCommandQueue(cqCommandQueue);
            if(cxGPUContext)
                clReleaseContext(cxGPUContext);
    }

    return EXIT_SUCCESS;
}

