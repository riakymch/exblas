/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// standard utilities and systems includes
#include "common.hpp"

#define NUM_ITER  10
////////////////////////////////////////////////////////////////////////////////
// Variables used in the program 
////////////////////////////////////////////////////////////////////////////////
cl_platform_id    cpPlatform;        //OpenCL platform
cl_device_id      cdDevice;          //OpenCL device list
cl_context        cxGPUContext;      //OpenCL context
cl_command_queue  cqCommandQueue;    //OpenCL command que
cl_mem            d_a, d_b;          //OpenCL memory buffer objects
cl_mem            d_x;
void              *h_A, *h_b;
void              *h_res;
double            *trsv_cpu;

static uint __n     = 0;
static uint __range = 0;
static uint __nbfpe = 0;
static uint __alg   = 0;

static void __usage(int argc __attribute__((unused)), char **argv) {
    fprintf(stderr, "Usage: %s [-n nbrows of A,\n", argv[0]);
    printf("                          -r range,\n");
    printf("                          -e nbfpe,\n");
    printf("                          -a alg (0-trsv, 1-acc, 2-fpe, 3-fpeex4, 4-fpeex6, 5-fpeex8)] \n");
    printf("       -?, -h:    Display this help and exit\n");
}

static void __parse_args(int argc, char **argv) {
    int i;

    for (i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-n") == 0)) {
             __n = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-r") == 0)) {
            __range = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-e") == 0)) {
            __nbfpe = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-a") == 0)) {
            __alg = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-h") || strcmp(argv[i], "-?")) == 0) {
            __usage(argc, argv);
            exit(-1);
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option %s\n", argv[i]);
            __usage(argc, argv);
            exit(-1);
        }
    }

    if (__n <= 0) {
        __usage(argc, argv);
        exit(-1);
    }
    if (__alg > 5) {
        __usage(argc, argv);
        exit(-1);
    }
}

int cleanUp (
    int exitCode
);

////////////////////////////////////////////////////////////////////////////////
//Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    __parse_args(argc, argv);
    printf("Starting with a matrix with  %i rows\n\n", __n);

    if (__alg == 0) {
        //runTRSV("../src/TRSV.Superacc.local.cl");
        runTRSV("../src/TRSV.cl");
    } else if (__alg == 1) {
        runTRSV("../src/TRSV.Superacc.cl");
    } else if (__alg == 2) {
        runTRSV("../src/TRSV.FPE.cl");
    } else if (__alg == 3) {
        __nbfpe = 4;
        runTRSV("../src/TRSV.FPE.EX.4.cl");
    } else if (__alg == 4) {
        __nbfpe = 6;
        runTRSV("../src/TRSV.FPE.EX.6.cl");
    } else if (__alg == 5) {
        __nbfpe = 8;
        runTRSV("../src/TRSV.FPE.EX.8.cl");
    }
}

int runTRSV(const char* program_file){
    cl_int ciErrNum;
    bool PassFailFlag = false;

    printf("Initializing data...\n");
        h_A = (double *) calloc(__n * __n, sizeof(double));
        h_b = (double *) calloc(__n, sizeof(double));
        h_res = (double *) malloc(__n * sizeof(double));
        trsv_cpu = (double *) malloc(__n * sizeof(double));

        // init data
        int emax = E_BITS - log2(__n);// use log in order to stay within [emin, emax]
        init_fpuniform_lu_matrix((double *) h_A, __n, __range, emax);
        init_fpuniform((double *) h_b, __n, __range, emax);
        //upper is row-wise
        //double c = 10;
        //int is_lower_column_wise = 1;
        //generate_ill_cond_system(is_lower_column_wise, (double *) h_A, (double *) h_b, __n, c);
        //printMatrix(is_lower_column_wise, (double *) h_A, __n, __n);
        //printVector((double *) h_b, __n);
        for (uint i = 0; i < __n; i++)
            trsv_cpu[i] = ((double *) h_b)[i];

    printf("Initializing OpenCL...\n");
        char platform_name[64];
#ifdef AMD
        strcpy(platform_name, "AMD Accelerated Parallel Processing");
#else
        strcpy(platform_name, "NVIDIA CUDA");
#endif
        //setenv("CUDA_CACHE_DISABLE", "1", 1);
        cpPlatform = GetOCLPlatform(platform_name);
        if (cpPlatform == NULL) {
            printf("ERROR: Failed to find the platform '%s' ...\n", platform_name);
            return -1;
        }

        //Get a GPU device
        cdDevice = GetOCLDevice(cpPlatform);
        if (cdDevice == NULL) {
            printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return -1;
        }

        //Create the context
        cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }

        //Create a command-queue
        cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }

    printf("Allocating OpenCL memory...\n\n");
        size_t size = __n * __n * sizeof(cl_double);
        d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_A, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        size = __n * sizeof(cl_double);
        d_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_b, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_b, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        d_x = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_x, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }

        printf("Initializing OpenCL TRSV...\n");
            ciErrNum = initTRSV(cxGPUContext, cqCommandQueue, cdDevice, program_file, __n, __alg, __nbfpe);

            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL TRSV with %u rows...\n\n", __n);
            //Just a single launch or a warmup iteration
            TRSV(NULL, d_x, d_a, d_b, __n, &ciErrNum);

            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);
#ifdef GPU_PROFILING
        double gpuTime[NUM_ITER];
        cl_event startMark, endMark;

        for(uint iter = 0; iter < NUM_ITER; iter++) {
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
            ciErrNum |= clFinish(cqCommandQueue);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueMarker, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                cleanUp(EXIT_FAILURE);
            }

            TRSV(NULL, d_x, d_a, d_b, __n, &ciErrNum);

            ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
            ciErrNum |= clFinish(cqCommandQueue);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueMarker, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                cleanUp(EXIT_FAILURE);
            }

            //Get OpenCL profiler time
            cl_ulong startTime = 0, endTime = 0;
            ciErrNum  = clGetEventProfilingInfo(startMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &startTime, NULL);
            ciErrNum |= clGetEventProfilingInfo(endMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clGetEventProfilingInfo Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                cleanUp(EXIT_FAILURE);
            }
            gpuTime[iter] = 1.0e-9 * ((unsigned long)endTime - (unsigned long)startTime); // / (double)NUM_ITER;
        }

        double minTime = min(gpuTime, NUM_ITER);
        //double throughput = __n * __n * sizeof(double);
        //throughput = (throughput / minTime) * 1e-9;
        double perf = __n * __n;
        perf = (perf / minTime) * 1e-9;
        //printf("Alg = %u \t NbFPE = %u \t Range = %u \t NbRows = %u \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", __alg, __nbfpe, __range, __n, minTime, throughput);
        printf("Alg = %u \t NbFPE = %u \t Range = %u \t NbRows = %u \t Time = %.8f s \t Performance = %.4f GFLOPS\n\n", __alg, __nbfpe, __range, __n, minTime, perf);
#endif

        printf("Validating TRSV OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_x, CL_TRUE, 0, __n * sizeof(cl_double), h_res, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
                //printMatrix(1, (const double *) h_A, __n, __n);
                //printVector((const double *) h_b, __n);
                //printVector((const double *) h_res, __n);
                /*PassFailFlag = verifyTRSVLNU((const double *) h_A, (const double *) h_b, (const double *) h_res, (const int) __n, 1e-15);
                if (PassFailFlag)
                    printf(" ...results on GPU are VERIFIED\n");
                else
                    printf(" ...results on GPU do NOT match\n");*/

            printf(" ...TRSV on CPU\n");
                TRSVLNU((double *) trsv_cpu, (const double *)h_A, __n);
                //printVector((const double *) trsv_cpu, __n);

            printf(" ...comparing the results\n");
                //printf("//--------------------------------------------------------\n");
                PassFailFlag &= compare((const double *) trsv_cpu, (const double *) h_res, __n, 1e-15);
                //PassFailFlag = compareTRSVLNUToMPFR((const double *)h_A, (double *) h_b, (double *) h_res, __n, 1e-15);

        //Release kernels and program
        printf("Shutting down...\n\n");
            closeTRSV();

    // pass or fail
    if (PassFailFlag)
        printf("[TRSV] test results...\tPASSED\n\n");
    else
        printf("[TRSV] test results...\tFAILED\n\n");

    return cleanUp(EXIT_SUCCESS);
}

int cleanUp (int exitCode) {
    //Release other OpenCL Objects
    if(d_a)
        clReleaseMemObject(d_a);
    if(d_b)
        clReleaseMemObject(d_b);
    if(d_x)
        clReleaseMemObject(d_x);
    if(cqCommandQueue)
        clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)
        clReleaseContext(cxGPUContext);

    //Release host buffers
    free(h_A);
    free(h_b);
    free(h_res);
    free(trsv_cpu);

    return exitCode;
}

