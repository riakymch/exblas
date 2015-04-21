/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cstring>

#include "config.h"
#include "ExSUM.hpp"
#include "blas1.hpp"
#include "ExSUM.Launcher.hpp"

#define NUM_ITER 20

#ifdef EXBLAS_TIMING
#include <cassert>

double min(double arr[], int size) {
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


/*
 * Parallel summation using our algorithm
 * If fpe < 2, use superaccumulators only,
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
double exsum(int N, double *a, int inca, int fpe, bool early_exit) {
    char path[256];
    strcpy(path, EXBLAS_BINARY_DIR);
    strcat(path, "/include/cl/");

    // with superaccumulators only
    if (fpe < 2) {
    //    return runExSUM(N, a, inca, 0, strcat(path, "ExSUM.Superacc.cl"));
        printf("Please use the size of FPE from this range [2, 8]\n");
        exit(0);
    }
    // there is no need and no improvement at all in using the early-exit technique for FPE of size 2
    if (fpe == 2)
        return runExSUM(N, a, inca, 2, strcat(path, "ExSUM.FPE.cl"));

    if (early_exit) {
        if (fpe <= 4)
            return runExSUM(N, a, inca, 4, strcat(path, "ExSUM.FPE.EX.4.cl"));
        if (fpe <= 6)
            return runExSUM(N, a, inca, 6, strcat(path, "ExSUM.FPE.EX.6.cl"));
        if (fpe <= 8)
            return runExSUM(N, a, inca, 8, strcat(path, "ExSUM.FPE.EX.8.cl"));
    } else { // ! early_exit
        if (fpe == 3)
            return runExSUM(N, a, inca, 3, strcat(path, "ExSUM.FPE.cl"));
        if (fpe == 4)
            return runExSUM(N, a, inca, 4, strcat(path, "ExSUM.FPE.cl"));
        if (fpe == 5)
            return runExSUM(N, a, inca, 5, strcat(path, "ExSUM.FPE.cl"));
        if (fpe == 6)
            return runExSUM(N, a, inca, 6, strcat(path, "ExSUM.FPE.cl"));
        if (fpe == 7)
            return runExSUM(N, a, inca, 7, strcat(path, "ExSUM.FPE.cl"));
        if (fpe == 8)
            return runExSUM(N, a, inca, 8, strcat(path, "ExSUM.FPE.cl"));
    }

    return 0.0;
}

double runExSUM(int N, double *h_a, int inca, int fpe, const char* program_file){
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
            ExSUM(NULL, d_Res, d_a, &ciErrNum);
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

            ExSUM(NULL, d_Res, d_a, &ciErrNum);

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
        printf("NbFPE = %u \t NbElements = %u \t \t Time = %.8f s \t Throughput = %.4f GB/s\n",
          fpe, N, minTime, ((1e-9 * N * sizeof(double)) / minTime));
#endif

        //Retrieving results...
            ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Res, CL_TRUE, 0, sizeof(cl_double), &h_Res, 0, NULL, NULL);
            if (ciErrNum != CL_SUCCESS) {
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

