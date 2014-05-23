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
Matrix            d_A, d_B, d_C;                //OpenCL memory buffer objects
double            *A, *B, *C;

static uint __nbRowsC    = 0;
static uint __nbColumnsC = 0;
static uint __nbRowsB    = 0;
static uint __range      = 0;
static uint __nbfpe      = 0;
static uint __alg        = 0;

static void __usage(int argc __attribute__((unused)), char **argv) {
  fprintf(stderr, "Usage: %s [-m nbrows of C -n nbcolumns of C -k nbcolumns of B\n -r range -e nbfpe\n -a alg (0-mine, 1-amd, 2-nvidia, 3-sapr, 4-fpepr, 5-salo, 6-fpelo, 7-sagl, 8-fpegl)] \n", argv[0]);
  printf("       -?, -h:    Display this help and exit\n");
}

static void __parse_args(int argc, char **argv) {
  int i;

  for (i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-m") == 0)) {
      __nbRowsC = atoi(argv[++i]);
    }if ((strcmp(argv[i], "-n") == 0)) {
      __nbColumnsC = atoi(argv[++i]);
    }if ((strcmp(argv[i], "-k") == 0)) {
      __nbRowsB = atoi(argv[++i]);
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

  if ((__nbRowsC <= 0) || (__nbColumnsC <= 0) || (__nbRowsB <= 0)) {
    __usage(argc, argv);
    exit(-1);
  }
  if (__alg > 8) {
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
    printf("Starting with a matrices of %ix%ix%i double elements\n\n", __nbRowsC, __nbRowsB, __nbColumnsC); 

    if (__alg == 0)
        runDGEMM("../src/DGEMM.cl");
    if (__alg == 1)
        runDGEMM("../src/DGEMM.AMD.cl");
    if (__alg == 2)
        runDGEMM("../src/DGEMM.NVIDIA.cl");
    if (__alg == 3)
        runDGEMM("../src/DGEMM.NVIDIA.Superacc.Private.cl");
    if (__alg == 4)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Private.cl");
    if (__alg == 5)
        runDGEMM("../src/DGEMM.NVIDIA.Superacc.Local.cl");
    if (__alg == 6)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Local.cl");
    if (__alg == 7)
        runDGEMM("../src/DGEMM.NVIDIA.Superacc.Global.cl");
    if (__alg == 8)
        runDGEMM("../src/DGEMM.NVIDIA.FPE.Global.cl");
}

int runDGEMM(const char* program_file){
    cl_int ciErrNum;
    int PassFailFlag = 1;
    int nbElements = 0;

    printf("Initializing data...\n");
        PassFailFlag  = posix_memalign((void **)&A, 64, __nbRowsC * __nbRowsB * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&B, 64, __nbRowsB * __nbColumnsC * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&C, 64, __nbRowsC * __nbColumnsC * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }
	// init data
        int emax = E_BITS - log2(__nbRowsC * __nbRowsB + __nbRowsB * __nbColumnsC + __nbRowsC * __nbColumnsC);// use log in order to stay within [emin, emax]
        init_fpuniform(A, __nbRowsC * __nbRowsB, __range, emax);
        init_fpuniform(B, __nbRowsB * __nbColumnsC, __range, emax);

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
	d_A.width = d_A.stride = __nbRowsB;
	d_A.height = __nbRowsC;
	size_t size = d_A.width * d_A.height * sizeof(double);
	d_A.elements = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, A, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_A, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	Matrix d_B;
	d_B.width = d_B.stride = __nbColumnsC;
	d_B.height = __nbRowsB;
	size = d_B.width * d_B.height * sizeof(double);
	d_B.elements = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, B, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_B, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	Matrix d_C;
	d_C.width = d_C.stride = __nbColumnsC;
	d_C.height = __nbRowsC;
	size = d_C.width * d_C.height * sizeof(double);
	d_C.elements = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_C, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    {
        printf("Initializing OpenCL DGEMM...\n");
	    if (__alg == 0)
                ciErrNum = initDGEMM(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            else if (__alg == 1)
                ciErrNum = initDGEMMAMD(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            else if (__alg == 2)
                ciErrNum = initDGEMMNVIDIA(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            else if (__alg == 3)
                ciErrNum = initDGEMMNVIDIAPrivate(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __nbColumnsC, __nbRowsC);
            else if (__alg == 4)
                ciErrNum = initDGEMMNVIDIAPrivate(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __nbColumnsC, __nbRowsC);
            else if (__alg == 5)
                ciErrNum = initDGEMMNVIDIAPrivate(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __nbColumnsC, __nbRowsC);
            else if (__alg == 6)
                ciErrNum = initDGEMMNVIDIAPrivate(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __nbColumnsC, __nbRowsC);
            else if (__alg == 7)
                ciErrNum = initDGEMMNVIDIAGlobal(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __nbColumnsC, __nbRowsC);
            else if (__alg == 8)
                ciErrNum = initDGEMMNVIDIAGlobal(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe, __nbColumnsC, __nbRowsC);
            
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

	nbElements = __nbRowsC * __nbRowsB + __nbRowsB * __nbColumnsC + __nbRowsC * __nbColumnsC;
        printf("Running OpenCL DGEMM with %u elements...\n\n", nbElements);
            //Just a single launch or a warmup iteration
            if (__alg == 0)
                DGEMM(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 1)
                DGEMMAMD(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 2)
                DGEMMNVIDIA(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 3)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 4)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 5)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 6)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 7)
                DGEMMNVIDIAGlobal(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 8)
                DGEMMNVIDIAGlobal(NULL, d_C, d_A, d_B, &ciErrNum);

            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

#ifdef GPU_PROFILING
	double gpuTime[NUM_ITER];
        cl_event startMark, endMark;

        for(uint iter = 0; iter < NUM_ITER; iter++) {
            ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
            ciErrNum |= clFinish(cqCommandQueue);
            //if (ciErrNum == CL_INVALID_COMMAND_QUEUE)
 	    //    printf("CL_INVALID_COMMAND_QUEUE\n");
            if (ciErrNum != CL_SUCCESS) {
                printf("Error in clEnqueueMarker, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                cleanUp(EXIT_FAILURE);
            }

            if (__alg == 0)
                DGEMM(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 1)
                DGEMMAMD(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 2)
                DGEMMNVIDIA(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 3)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 4)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 5)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 6)
                DGEMMNVIDIAPrivate(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 7)
                DGEMMNVIDIAGlobal(NULL, d_C, d_A, d_B, &ciErrNum);
            else if (__alg == 8)
                DGEMMNVIDIAGlobal(NULL, d_C, d_A, d_B, &ciErrNum);

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
	perf = 2.0 * d_A.width * d_B.width * d_A.height;
	perf = (perf / minTime) * 1e-9;
        printf("Alg = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", __alg, __range, nbElements, nbElements * sizeof(double), minTime, throughput);
        printf("Alg = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Performance = %.4f GFLOPS\n\n", __alg, __range, nbElements, nbElements * sizeof(double), minTime, perf);
#endif

        printf("Validating DGEMM OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_C.elements, CL_TRUE, 0, d_C.width * d_C.height * sizeof(double), C, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }

            printf(" ...DGEMM on CPU\n");
		double *C_CPU;
                C_CPU = (double *) calloc(__nbRowsC * __nbColumnsC, sizeof(double));
                DGEMMCPU(C_CPU, (const double *)A, (const double *)B, __nbRowsC, __nbColumnsC, __nbRowsB);
                //printMatrix(C, d_C.width, d_C.height);
                //printMatrix(C_CPU, d_C.width, d_C.height);

            printf(" ...comparing the results\n");
                printf("//--------------------------------------------------------\n");
		//Compare the GPU to the CPU results
		PassFailFlag = compare((const double *) C_CPU, (const double *) C, __nbRowsC * __nbColumnsC, 1e-16);
                 
		//PassFailFlag = compareDGEMMWithMPFR((const double *)C_CPU, (const double *)A, (const double *)B, __nbRowsC, __nbColumnsC, __nbRowsB);
                printf("//--------------------------------------------------------\n");
		free(C_CPU);
		
         //Release kernels and program
         printf("Shutting down...\n\n");
            if (__alg == 0)
		closeDGEMM();
            else if (__alg == 1)
                closeDGEMMAMD();
            else if (__alg == 2)
                closeDGEMMNVIDIA();
            else if (__alg == 3)
                closeDGEMMNVIDIAPrivate();
            else if (__alg == 4)
                closeDGEMMNVIDIAPrivate();
            else if (__alg == 5)
                closeDGEMMNVIDIAPrivate();
            else if (__alg == 6)
                closeDGEMMNVIDIAPrivate();
            else if (__alg == 7)
                closeDGEMMNVIDIAGlobal();
            else if (__alg == 8)
                closeDGEMMNVIDIAGlobal();
    }

    // pass or fail
    if (PassFailFlag)
	printf("[DGEMM] test results...\tPASSED\n");
    else
	printf("[DGEMM] test results...\tFAILED\n");

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
    free(A);
    free(B);
    free(C);
    
    return exitCode;
}

