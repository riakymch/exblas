/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file gpu/blas1/ExDOT.cpp
 *  \brief Provides implementations of a set of dot routines
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
#include "blas1.hpp"
#include "ExDOT.Launcher.hpp"


#define NUM_ITER 20


/**
 * \ingroup ExDOT
 * \brief Executes on GPU parallel dot product of two real vectors.
 *     For internal use
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param offseta specifies position in the vector a from its start
 * \param bg vector
 * \param incb specifies the increment for the elements of b
 * \param offsetb specifies position in the vector b from its start
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return Contains the reproducible and accurate dot product of two real vectors
 */
static double runExDOT(const int N, double *h_a, const int inca, const int offseta, double *h_b, const int incb, const int offsetb, const int fpe, const char* program_file);

/**
 * \ingroup ExDOT
 * \brief Parallel dot forms the dot product of two vectors with our
 *     multi-level reproducible and accurate algorithm.
 *
 *     If fpe < 3, it uses superaccumulators only. Otherwise, it relies on 
 *     floating-point expansions of size FPE with superaccumulators when needed
 *
 * \param Ng vector size
 * \param ag vector
 * \param inca specifies the increment for the elements of a
 * \param offseta specifies position in the vector a from its start
 * \param bg vector
 * \param incb specifies the increment for the elements of b
 * \param offsetb specifies position in the vector b from its start
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate result of the dot product of two real vectors
 */
double exdot(const int Ng, double *ag, const int inca, const int offseta, double *bg, const int incb, const int offsetb, const int fpe, const bool early_exit) {
    if (Ng <= 0)
        return 0.0;

    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // with superaccumulators only
    if (fpe < 3)
        return runExDOT(Ng, ag, inca, offseta, bg, incb, offsetb, 0, strcat(path, "ExDOT.Superacc.cl"));

    if (early_exit) {
        if (fpe <= 4)
            return runExDOT(Ng, ag, inca, offseta, bg, incb, offsetb, 4, strcat(path, "ExDOT.FPE.EX.4.cl"));
        if (fpe <= 6)
            return runExDOT(Ng, ag, inca, offseta, bg, incb, offsetb, 6, strcat(path, "ExDOT.FPE.EX.6.cl"));
        if (fpe <= 8)
            return runExDOT(Ng, ag, inca, offseta, bg, incb, offsetb, 8, strcat(path, "ExDOT.FPE.EX.8.cl"));
    } else // ! early_exit
        return runExDOT(Ng, ag, inca, offseta, bg, incb, offsetb, fpe, strcat(path, "ExDOT.FPE.cl"));

    return 0.0;
}

static double runExDOT(const int N, double *h_a, const int inca, const int offseta, double *h_b, const int incb, const int offsetb, const int fpe, const char* program_file){
    double h_Res;
    cl_int ciErrNum;

    // Initializing OpenCL
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
        cl_mem d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), h_a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("N = %d\t ciErrNum = %d\n", N, ciErrNum);
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), h_b, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_b, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_Res = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_res, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

    {
        //Initializing OpenCL ExDOT
            ciErrNum = initExDOT(cxGPUContext, cqCommandQueue, cdDevice, program_file, fpe);
            if (ciErrNum != CL_SUCCESS)
                exit(EXIT_FAILURE);

        //Running OpenCL ExDOT
            //Just a single launch or a warmup iteration
            ExDOT(N, d_a, inca, offseta, d_b, incb, offsetb, NULL, d_Res, &ciErrNum);
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

            ExDOT(N, d_a, inca, offseta, d_b, incb, offsetb, NULL, d_Res, &ciErrNum);

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

        printf("NbFPE = %u \t NbElements = %u \t \t Time = %.8f s \t Throughput = %.4f GB/s\n", fpe, N, mint, ((1e-9 * 2 * N * sizeof(double)) / mint));
        fprintf(stderr, "%f ", mint);
#endif

        //Retrieving results
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Res, CL_TRUE, 0, sizeof(cl_double), &h_Res, 0, NULL, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

         //Release kernels and program
         //Shutting down and freeing memory
            closeExDOT();
            if(d_a)
                clReleaseMemObject(d_a);
            if(d_b)
                clReleaseMemObject(d_b);
            if(d_Res)
                clReleaseMemObject(d_Res);
            if(cqCommandQueue)
                clReleaseCommandQueue(cqCommandQueue);
            if(cxGPUContext)
                clReleaseContext(cxGPUContext);
    }

    return h_Res;
}

