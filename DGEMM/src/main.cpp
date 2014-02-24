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

#define NUM_ITER  40
////////////////////////////////////////////////////////////////////////////////
// Variables used in the program 
////////////////////////////////////////////////////////////////////////////////
cl_platform_id    cpPlatform;                   //OpenCL platform
cl_device_id      cdDevice;                     //OpenCL device list    
cl_context        cxGPUContext;                 //OpenCL context
cl_command_queue  cqCommandQueue;               //OpenCL command que
Matrix            d_A, d_B, d_C;                //OpenCL memory buffer objects
Matrix            A, B, C;

static uint __mC    = 0;
static uint __nB    = 0;
static uint __kC    = 0;
static uint __range = 0;
static uint __nbfpe = 0;
static uint __alg   = 0;

static void __usage(int argc __attribute__((unused)), char **argv) {
  fprintf(stderr, "Usage: %s [-m number of rows in A -n number of columns in A -k number of columns in B -r range -e nbfpe -a alg (0-dgemm)] \n", argv[0]);
  printf("       -?, -h:    Display this help and exit\n");
}

static void __parse_args(int argc, char **argv) {
  int i;

  for (i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-m") == 0)) {
      __mC = atoi(argv[++i]);
    }if ((strcmp(argv[i], "-n") == 0)) {
      __nB = atoi(argv[++i]);
    }if ((strcmp(argv[i], "-k") == 0)) {
      __kC = atoi(argv[++i]);
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

  if ((__mC <= 0) || (__nB <= 0) || (__kC <= 0)) {
    __usage(argc, argv);
    exit(-1);
  }
  if (__alg > 1) {
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
    printf("Starting with a matrices of %ix%ix%i double elements\n\n", __mC, __nB, __kC); 

    if (__alg == 0)
        runDGEMM("../src/DGEMM.cl");
}

int runDGEMM(const char* program_file){
    cl_int ciErrNum;
    int    PassFailFlag = 1;
    int nbElements = 0;

    printf("Initializing data...\n");
	A.width = A.stride = __mC;
	A.height = __nB;
	B.width = B.stride = __nB;
	B.height = __kC;
	C.width = C.stride = __mC;
	C.height = __kC;
        PassFailFlag  = posix_memalign((void **)&A.elements, 64, A.width * A.height * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&B.elements, 64, B.width * B.height * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&C.elements, 64, C.width * C.height * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }
	// init data
        int emax = E_BITS - log2(A.width * A.height + B.width * B.height + C.width * C.height);// use log in order to stay within [emin, emax]
        init_fpuniform((double *) A.elements, A.width * A.height, __range, emax);
        init_fpuniform((double *) B.elements, B.width * B.height, __range, emax);
        init_fpuniform((double *) C.elements, C.width * C.height, __range, emax);

    printf("Initializing OpenCL...\n");
        char platform_name[64];
	char device_name[32];
#ifdef AMD
        strcpy(platform_name, "AMD Accelerated Parallel Processing");
	strcpy(device_name, "Tahiti");
#else
        strcpy(platform_name, "NVIDIA CUDA");
        strcpy(device_name, "Tesla K20c");
#endif
        //setenv("CUDA_CACHE_DISABLE", "1", 1);
        cpPlatform = GetOCLPlatform(platform_name);
        if (cpPlatform == NULL) {
            printf("ERROR: Failed to find the platform '%s' ...\n", platform_name);
            return -1;
        }

        //Get a GPU device
        cdDevice = GetOCLDevice(cpPlatform, device_name);
        if (cdDevice == NULL) {
            printf("ERROR: Failed to find the device '%s' ...\n", device_name);
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
	Matrix d_A;
	d_A.width = d_A.stride = A.width;
	d_A.height = A.height;
	size_t size = d_A.width * d_A.height * sizeof(double);
	d_A.elements = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, A.elements, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_A, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	Matrix d_B;
	d_B.width = d_B.stride = B.width;
	d_B.height = B.height;
	size = d_B.width * d_B.height * sizeof(double);
	d_B.elements = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, B.elements, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_B, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	Matrix d_C;
	d_C.width = d_C.stride = C.width;
	d_C.height = C.height;
	size = d_C.width * d_C.height * sizeof(double);
	d_C.elements = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_C, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    {
        printf("Initializing OpenCL DGEMM...\n");
            ciErrNum = initDGEMM(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

	nbElements = A.width * A.height + B.width * B.height + C.width * C.height;
        printf("Running OpenCL DGEMM with %u elements...\n\n", nbElements);
            //Just a single launch or a warmup iteration
            DGEMM(NULL, d_C, d_A, d_B, &ciErrNum);
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

            DGEMM(NULL, d_C, d_A, d_B, &ciErrNum);

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
	double perf;
	perf = 1.0 / minTime;	
	perf *= 1e-9;
	perf *= nbElements * sizeof(double);
        printf("Alg = 0 \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", 
            __range, nbElements, nbElements * sizeof(double), minTime, perf);
	perf = 1. / minTime;
	perf *= 2.0 * C.width * C.height * B.width;
        printf("Alg = 0 \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Performance = %.4f GFLOPS\n\n", 
            __range, nbElements, nbElements * sizeof(double), minTime, perf);
#endif

        printf("Validating DGEMM OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_C.elements, CL_TRUE, 0, C.width * C.height * sizeof(double), C.elements, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
            //Release kernels and program
         printf("Shutting down...\n\n");
            closeDGEMM();
    }

    // pass or fail
    if (!PassFailFlag)
	printf("[DGEMM] test results...\nPASSED\n");
    else
	printf("[DGEMM] test results...\nFAILED\n");

    cleanUp(EXIT_SUCCESS);
}

int cleanUp (int exitCode) {
    //Release other OpenCL Objects
    if(d_A.elements) 
	clReleaseMemObject(d_A.elements);
    if(d_B.elements) 
	clReleaseMemObject(d_B.elements);
    if(d_C.elements) 
	clReleaseMemObject(d_C.elements);
    if(cqCommandQueue) 
	clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext) 
	clReleaseContext(cxGPUContext);

    //Release host buffers
    free(A.elements);
    free(B.elements);
    free(C.elements);
    
    return exitCode;
}

