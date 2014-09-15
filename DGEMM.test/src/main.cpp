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

#define NUM_ITER  20
////////////////////////////////////////////////////////////////////////////////
// Variables used in the program 
////////////////////////////////////////////////////////////////////////////////
cl_platform_id    cpPlatform;                   //OpenCL platform
cl_device_id      cdDevice;                     //OpenCL device list    
cl_context        cxGPUContext;                 //OpenCL context
cl_command_queue  cqCommandQueue;               //OpenCL command que
cl_mem            d_A, d_B, d_C;                //OpenCL memory buffer objects
double            *A, *B, *C;

static uint __m     = 0;
static uint __n     = 0;
static uint __k     = 0;
static uint __range = 0;
static uint __nbfpe = 0;
static uint __alg   = 0;
static uint __multi = 1;

static void __usage(int argc __attribute__((unused)), char **argv) {
    fprintf(stderr, "Usage:\n  %s [-m nbrows of C,\n", argv[0]);
    printf("              -n nbcolumns of C,\n");
    printf("              -k nbrows of B,\n");
    printf("              -r range,\n");
    printf("              -e nbfpe,\n");
    printf("              -a alg (0-mine, 1-amd, 2-nvidia, 30-pr-sa, 31-pr-fpe, 32-pr-fpe-ex-4, 33-pr-fpe-ex-6, 34-pr-fpe-ex-8, 40-lo-sa, 41-lo-fpe, 50-gl-sa, 51-gl-fpe, 52-gl-fpe-ex-4, 53-gl-fpe-ex-8, 54-gl-fpe-multi, 6-volkov),\n");
    printf("              -ml multi-values] \n");
    printf("  -?, -h:    Display this help and exit\n");
}

static void __parse_args(int argc, char **argv) {
    int i;

    for (i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0)) {
            __m = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-n") == 0)) {
           __n = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-k") == 0)) {
            __k = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-r") == 0)) {
            __range = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-e") == 0)) {
            __nbfpe = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-a") == 0)) {
            __alg = atoi(argv[++i]);
        } if ((strcmp(argv[i], "-ml") == 0)) {
            __multi = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-h") || strcmp(argv[i], "-?")) == 0) {
            __usage(argc, argv);
            exit(-1);
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "Unknown option %s\n", argv[i]);
            __usage(argc, argv);
            exit(-1);
        }
    }

    if ((__m <= 0) || (__n <= 0) || (__k <= 0)) {
        __usage(argc, argv);
        exit(-1);
    }

    uint algs[]= {0,1,2,30,31,32,33,34,40,41,50,51,52,53,54,6};
    int is_alg = 0;
    for (i = 0; i < 1666666; i++)
        if (algs[i] == __alg) {
            is_alg = 1;
            break;
        }
    if (!is_alg) {
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
    printf("Starting with a matrices of %ix%ix%i double\n\n", __m, __k, __n); 

    if (__alg == 0)
        runDGEMM("../src/DGEMM.cl");
    else if (__alg == 1)
        runDGEMM("../src/DGEMM.AMD.cl");
    else if (__alg == 2)
        runDGEMM("../src/DGEMM.NVIDIA.cl");
    else if (__alg == 30)
        runDGEMM("../src/DGEMM.NVIDIA.Superacc.Private.cl");
    else if (__alg == 31)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Private.cl");
    else if (__alg == 32)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.EX4.Private.cl");
    else if (__alg == 33)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.EX6.Private.cl");
    else if (__alg == 34)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.EX8.Private.cl");
    else if (__alg == 40)
        runDGEMM("../src/DGEMM.NVIDIA.Superacc.Local.cl");
    else if (__alg == 41)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Local.cl");
    else if (__alg == 50)
        runDGEMM("../src/DGEMM.NVIDIA.Superacc.Global.cl");
    else if (__alg == 51)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Global.cl");
    else if (__alg == 52)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.EX4.Global.cl");
    else if (__alg == 53)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.EX8.Global.cl");
    else if (__alg == 54)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Multi.Global.cl");
    else if (__alg == 6)
        runDGEMM("../src/DGEMM.Volkov.cl");
}

int runDGEMM(const char* program_file){
    cl_int ciErrNum;
    int PassFailFlag = 1;
    int nbElements = 0;

    printf("Initializing data...\n");
        /*PassFailFlag  = posix_memalign((void **)&A, 64, __m * __k * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&B, 64, __k * __n * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&C, 64, __m * __n * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }*/
        A = (double *) malloc(__m * __k * sizeof(double));
        B = (double *) malloc(__k * __n * sizeof(double));
        C = (double *) malloc(__m * __n * sizeof(double));
        // init data
        nbElements = __m * __k + __k * __n + __m * __n;
        int emax = E_BITS - log2(nbElements);// use log in order to stay within [emin, emax]
        init_fpuniform(A, __m * __k, __range, emax);
        init_fpuniform(B, __k * __n, __range, emax);

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
        size_t size = __m * __k * sizeof(double);
        d_A = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, A, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_A, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        size = __k * __n * sizeof(double);
        d_B = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, B, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_B, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        size = __m * __n * sizeof(double);
        d_C = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_C, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }

        printf("Initializing OpenCL DGEMM...\n");
            if (__alg == 0)
                ciErrNum = initDGEMM(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            else if (__alg == 1)
                ciErrNum = initDGEMMAMD(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            else if (__alg == 2)
                ciErrNum = initDGEMMNVIDIA(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            else if (((__alg >= 30) && (__alg <= 34)) || (__alg == 40) || (__alg == 41))
                ciErrNum = initDGEMMNVIDIAPrivate(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe);
            else if ((__alg >= 50) && (__alg <= 54))
                ciErrNum = initDGEMMNVIDIAGlobal(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __m, __n, __multi);
            else if (__alg == 6)
                ciErrNum = initDGEMMVolkov(cxGPUContext, cqCommandQueue, cdDevice, program_file);

            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL DGEMM with %u...\n\n", nbElements);
            //Just a single launch or a warmup iteration
            if (__alg == 0)
                DGEMM(NULL, d_C, d_A, d_B, __m, __n, __k, &ciErrNum);
            else if (__alg == 1)
                DGEMMAMD(NULL, d_C, d_A, d_B, __m, &ciErrNum);
            else if (__alg == 2)
                DGEMMNVIDIA(NULL, d_C, d_A, d_B, __m, __n, &ciErrNum);
            else if (((__alg >= 30) && (__alg <= 34)) || (__alg == 40) || (__alg == 41))
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, __m, __n, __multi, &ciErrNum);
            else if ((__alg >= 50) && (__alg <= 54))
                DGEMMNVIDIAGlobal(NULL, d_C, d_A, d_B, __m, __n, __multi, &ciErrNum);
            else if (__alg == 6)
                DGEMMVolkov(NULL, d_C, d_A, d_B, __m, __n, __k, &ciErrNum);

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

            if (__alg == 0)
                DGEMM(NULL, d_C, d_A, d_B, __m, __n, __k, &ciErrNum);
            else if (__alg == 1)
                DGEMMAMD(NULL, d_C, d_A, d_B, __m, &ciErrNum);
            else if (__alg == 2)
                DGEMMNVIDIA(NULL, d_C, d_A, d_B, __m, __n, &ciErrNum);
            else if (((__alg >= 30) && (__alg <= 34)) || (__alg == 40) || (__alg == 41))
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, __m, __n, __multi, &ciErrNum);
            else if ((__alg >= 50) && (__alg <= 54))
                DGEMMNVIDIAGlobal(NULL, d_C, d_A, d_B, __m, __n, __multi, &ciErrNum);
            else if (__alg == 6)
                DGEMMVolkov(NULL, d_C, d_A, d_B, __m, __n, __k, &ciErrNum);

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
        double perf = nbElements * sizeof(double);
        double throughput = (perf / minTime) * 1e-9;
        if ((__alg <= 2) || (__alg == 6)) {
            perf = 2.0 * __m;
            perf *= __n * __k;
        } else if (__alg % 10 == 0) {
            perf = (2.0 + 8.0) * __m;
            perf *= __n * __k;
        } else if ((__alg == 31) || (__alg == 34) || (__alg == 34) || (__alg == 41) || (__alg == 51) || (__alg == 54)) {
            perf = (2.0 + 12.0 * __nbfpe) * __m;
            perf *= __n * __k;
        } else if ((__alg == 32) || (__alg == 52)) {
            perf = 50.0 * __m;
            perf *= __n * __k;
        } else if (__alg == 33) {
            perf = 74.0 * __m;
            perf *= __n * __k;
        } else if ((__alg == 34) || (__alg == 53)) {
            perf = 98.0 * __m;
            perf *= __n * __k;
        }
        perf = (perf / minTime) * 1e-9;
        //printf("Alg = %u \t NbFPE = %u \t Range = %u \t Size = %u \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", __alg, __nbfpe, __range, __n, minTime, throughput);
        printf("Alg = %u \t NbFPE = %u \t Range = %u \t Size = %u \t Time = %.8f s \t Performance = %.4f GFLOPS\n\n", __alg, __nbfpe, __range, __n, minTime, perf);
#endif

        printf("Validating DGEMM OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_C, CL_TRUE, 0, __m * __n * sizeof(double), C, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }

            printf(" ...DGEMM on CPU\n");
                /*double *C_CPU;
                C_CPU = (double *) calloc(__m * __n, sizeof(double));
                DGEMMCPU(C_CPU, (const double *)A, (const double *)B, __m, __n, __k);*/
                /*printMatrix(C, __m, __n);
                printf("\n");
                printMatrix(C_CPU, __m, __n);*/

            printf(" ...comparing the results\n");
                printf("//--------------------------------------------------------\n");
                //PassFailFlag = compare((const double *) C_CPU, (const double *) C, __m * __n, 1e-16);
                PassFailFlag = compareDGEMMWithMPFR((const double *)C, (const double *)A, (const double *)B, __m, __n, __k, 1e-16);
                printf("//--------------------------------------------------------\n");
                //free(C_CPU);

         //Release kernels and program
         printf("Shutting down...\n\n");
            if (__alg == 0)
                closeDGEMM();
            else if (__alg == 1)
                closeDGEMMAMD();
            else if (__alg == 2)
                closeDGEMMNVIDIA();
            else if (((__alg >= 30) && (__alg <= 34)) || (__alg == 40) || (__alg == 41))
                closeDGEMMNVIDIAPrivate();
            else if ((__alg >= 50) && (__alg <= 54))
                closeDGEMMNVIDIAGlobal();
            if (__alg == 6)
                closeDGEMMVolkov();

    // pass or fail
    if (PassFailFlag)
        printf("[DGEMM] test results...\tPASSED\n");
    else
        printf("[DGEMM] test results...\tFAILED\n");

    return cleanUp(EXIT_SUCCESS);
}

int cleanUp (int exitCode) {
    //Release other OpenCL Objects
    if(d_A) 
        clReleaseMemObject(d_A);
    if(d_B) 
        clReleaseMemObject(d_B);
    if(d_C) 
        clReleaseMemObject(d_C);
    if(cqCommandQueue) 
        clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext) 
        clReleaseContext(cxGPUContext);

    //Release host buffers
    free(A);
    free(B);
    free(C);

    return exitCode;
}

