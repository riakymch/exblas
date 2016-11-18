/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file gpu/blas1/ExSUM.cpp
 *  \brief Provides implementations of a set of sum routines
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
#include "ExSUM.Launcher.hpp"


#define NUM_ITER 20


/**
 * \ingroup ExSUM
 * \brief Executes on GPU parallel summation/reduction on elements of a real vector.
 *     For internal use
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector to start with 
 * \param fpe size of floating-point expansion
 * \param program_file path to the file with kernels
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
static double runExSUM(const int N, double *a, const int inca, const int offset, const int fpe, const char* program_file);

/**
 * \ingroup ExSUM
 * \brief Parallel summation computes the sum of elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm.
 *
 *     If fpe < 2, it uses superaccumulators only. Otherwise, it relies on 
 *     floating-point expansions of size FPE with superaccumulators when needed
 *
 * \param Ng vector size
 * \param ag vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector to start with 
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double exsum(const int Ng, double *ag, const int inca, const int offset, const int fpe, const bool early_exit) {
    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // with superaccumulators only
    if (fpe < 2)
        return runExSUM(Ng, ag, inca, offset, 0, strcat(path, "ExSUM.Superacc.cl"));

    if (early_exit) {
        if (fpe <= 4)
            return runExSUM(Ng, ag, inca, offset, 4, strcat(path, "ExSUM.FPE.EX.4.cl"));
        if (fpe <= 6)
            return runExSUM(Ng, ag, inca, offset, 6, strcat(path, "ExSUM.FPE.EX.6.cl"));
        if (fpe <= 8)
            return runExSUM(Ng, ag, inca, offset, 8, strcat(path, "ExSUM.FPE.EX.8.cl"));
    } else // ! early_exit
        return runExSUM(Ng, ag, inca, offset, fpe, strcat(path, "ExSUM.FPE.cl"));

    return 0.0;
}

static double runExSUM(const int N, double *h_a, const int inca, const int offset, const int fpe, const char* program_file){
    double h_Res;
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
            printf("Error = %d\n", ciErrNum);
            printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

        //Allocating OpenCL memory...
        cl_mem d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_double), h_a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }
        cl_mem d_Res = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_res, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            exit(EXIT_FAILURE);
        }

    {
        //Initializing OpenCL dSum...
            ciErrNum = initExSUM(cxGPUContext, cqCommandQueue, cdDevice, program_file, N, fpe);
            if (ciErrNum != CL_SUCCESS)
                exit(EXIT_FAILURE);

        //Running OpenCL dSum with %u elements...
            //Just a single launch or a warmup iteration
            ExSUM(NULL, d_Res, d_a, inca, offset, &ciErrNum);
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

            ExSUM(NULL, d_Res, d_a, inca, offset, &ciErrNum);

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
            mint = std::min(mint, t);
        }

        printf("NbFPE = %u \t NbElements = %u \t \t Time = %.8f s \t Throughput = %.4f GB/s\n", fpe, N, mint, ((1e-9 * N * sizeof(double)) / mint));
        fprintf(stderr, "%f ", mint);
#endif

        //Retrieving results...
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Res, CL_TRUE, 0, sizeof(cl_double), &h_Res, 0, NULL, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("ciErrNum = %d\n", ciErrNum);
                printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                exit(EXIT_FAILURE);
            }

         //Release kernels and program
         //Shutting down and freeing memory...
            closeExSUM();
            if(d_a)
                clReleaseMemObject(d_a);
            if(d_Res)
                clReleaseMemObject(d_Res);
            if(cqCommandQueue)
                clReleaseCommandQueue(cqCommandQueue);
            if(cxGPUContext)
                clReleaseContext(cxGPUContext);
    }

    return h_Res;
}

