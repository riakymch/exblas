/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
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


#define NUM_ITER 5


/**
 * \ingroup ExTRSV
 * \brief Executes on GPU parallel triangular solver A*x = b or A**T*x = b.
 *     For internal use
 *
 * \param n size of matrix A
 * \param a matrix A
 * \param lda leading dimension of A
 * \param offseta specifies position in the metrix A from its beginning
 * \param x vector
 * \param incx the increment for the elements of a
 * \param offsetx specifies position in the vector x from its start
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return Contains the reproducible and accurate result of solving triangular system
 */
int runExTRSV(const int n, double *a, const int lda, const int offseta, double *x, const int incx, const int offsetx, const int fpe, const char* program_file);

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
 * \param offseta specifies position in the metrix A from its beginning
 * \param x vector
 * \param incx the increment for the elements of a
 * \param offsetx specifies position in the vector x from its start
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return vector x contains the reproducible and accurate result of ExTRSV
 */
int extrsv(const char uplo, const char transa, const char diag, const int n, double *a, const int lda, const int offseta, double *x, const int incx, const int offsetx, const int fpe, const bool early_exit) {
    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // ExTRSV
    if (fpe == 0)
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.Superacc.cl") : strcat(path, "ExTRSV.unn.Superacc.cl"));

    // DTRSV
    if (fpe == 1)
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "DTRSV.lnn.cl") : strcat(path, "DTRSV.unn.cl"));

    if ((early_exit) && (fpe <= 8)) {
        if (fpe <= 4)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 4, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.4.cl") : strcat(path, "ExTRSV.unn.FPE.EX.4.cl"));
        if (fpe <= 6)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 6, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.6.cl") : strcat(path, "ExTRSV.unn.FPE.EX.6.cl"));
        if (fpe <= 8)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 8, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.8.cl") : strcat(path, "ExTRSV.unn.FPE.EX.8.cl"));
    } else if (fpe <= 8) // ! early_exit
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.cl") : strcat(path, "ExTRSV.unn.FPE.cl"));

    // ExTRSV with ExIR
    if (fpe == 10)
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.Superacc.IR.cl") : strcat(path, "ExTRSV.unn.Superacc.IR.cl"));

    if ((early_exit) && (fpe <= 18)) {
        if (fpe <= 14)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 14, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.4.IR.cl") : strcat(path, "ExTRSV.unn.FPE.EX.4.IR.cl"));
        if (fpe <= 16)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 16, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.6.IR.cl") : strcat(path, "ExTRSV.unn.FPE.EX.6.IR.cl"));
        if (fpe <= 18)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 18, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.EX.8.IR.cl") : strcat(path, "ExTRSV.unn.FPE.EX.8.IR.cl"));
    } else if (fpe <= 18) // ! early_exit
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "ExTRSV.lnn.FPE.IR.cl") : strcat(path, "ExTRSV.unn.FPE.IR.cl"));

    // DTRSV with ExIR
    if (fpe == 20)
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "DTRSV.lnn.cl") : strcat(path, "DTRSV.unn.cl"));
    if (fpe == 21)
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "DTRSV.lnn.ExIR.Superacc.cl") : strcat(path, "DTRSV.unn.ExIR.Superacc.cl"));

    if ((early_exit) && (fpe <= 28)) {
        if (fpe <= 24)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 24, (uplo == 'L') ? strcat(path, "DTRSV.lnn.ExIR.FPE.EX.4.cl") : strcat(path, "DTRSV.unn.ExIR.FPE.EX.4.cl"));
        if (fpe <= 26)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 26, (uplo == 'L') ? strcat(path, "DTRSV.lnn.ExIR.FPE.EX.6.cl") : strcat(path, "DTRSV.unn.ExIR.FPE.EX.6.cl"));
        if (fpe <= 28)
            return runExTRSV(n, a, lda, offseta, x, incx, offsetx, 28, (uplo == 'L') ? strcat(path, "DTRSV.lnn.ExIR.FPE.EX.8.cl") : strcat(path, "DTRSV.unn.ExIR.FPE.EX.8.cl"));
    } else if (fpe <= 28) // ! early_exit
        return runExTRSV(n, a, lda, offseta, x, incx, offsetx, fpe, (uplo == 'L') ? strcat(path, "DTRSV.lnn.ExIR.FPE.cl") : strcat(path, "DTRSV.unn.ExIR.FPE.cl"));

    return 0;
}

int runExTRSV(const int n, double *a, const int lda, const int offseta, double *x, const int incx, const int offsetx, const int fpe, const char* program_file) {
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
        if (fpe >= 10) {
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

        if ((fpe >= 10) && (fpe < 20)) {
            // ExTRSV w ExIR
            ExTRSVIR(NULL, n, d_a, lda, d_x, incx, d_b, &ciErrNum);
        } else if ((fpe >= 21) && (fpe < 30)) {
            // DTRSV w ExIR
            DTRSVExIR(NULL, n, d_a, lda, d_x, incx, d_b, &ciErrNum);
        } else {
            // ExTRSV
            ExTRSV(NULL, n, d_a, lda, offseta, d_x, incx, offsetx, &ciErrNum);
        }
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

            if ((fpe >= 10) && (fpe < 20)) {
                // ExTRSV w ExIR
                ExTRSVIR(NULL, n, d_a, lda, d_x, incx, d_b, &ciErrNum);
            } else if ((fpe >= 21) && (fpe < 30)) {
                // DTRSV w ExIR
                DTRSVExIR(NULL, n, d_a, lda, d_x, incx, d_b, &ciErrNum);
            } else {
                // ExTRSV
                ExTRSV(NULL, n, d_a, lda, offseta, d_x, incx, offsetx, &ciErrNum);
            }

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

        double perf = n * n;
        perf = (perf / mint) * 1e-9;
        printf("NbFPE = %u \t N = %u \t \t Time = %.8f s \t Performance = %.4f GFLOPS\n", fpe, n, mint, perf);
        fprintf(stderr, "%f  ", mint);
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
            //if(fpe == 1)
            //    clReleaseMemObject(d_b);
            clReleaseCommandQueue(cqCommandQueue);
            clReleaseContext(cxGPUContext);
    }

    return EXIT_SUCCESS;
}

