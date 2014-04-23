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
cl_platform_id    cpPlatform;            //OpenCL platform
cl_device_id      cdDevice;              //OpenCL device list    
cl_context        cxGPUContext;          //OpenCL context
cl_command_queue  cqCommandQueue;        //OpenCL command que
cl_mem            d_a, d_b, d_Superacc, d_res;  //OpenCL memory buffer objects
double            *h_a, *h_b, h_res;
bintype           *h_Superacc;

static uint __nbElements = 0;
static uint __range      = 0;
static uint __nbfpe      = 0;
static uint __alg        = 0;

static void __usage(int argc __attribute__((unused)), char **argv) {
  fprintf(stderr, "Usage: %s [-n number of elements -r range -e nbfpe -a alg (0-ddot, 1-acc, 2-fpe, 3-fpeee)] \n", argv[0]);
  printf("       -?, -h:    Display this help and exit\n");
}

static void __parse_args(int argc, char **argv) {
  int i;

  for (i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-n") == 0)) {
      __nbElements = atoi(argv[++i]);
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

  if (__nbElements <= 0) {
    __usage(argc, argv);
    exit(-1);
  }
  if (__alg > 3) {
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
    printf("Starting with two vectors with  %i double elements\n\n", __nbElements); 

    if (__alg == 0)
        runDDOTSimple("../src/DDOT.Simple.cl");
    if (__alg == 1)
        runDDOT("../src/DDOT.Superacc.cl");
    if (__alg == 2)
        runDDOT("../src/DDOT.FPE.cl");
    if (__alg == 3)
        runDDOT("../src/DDOT.FPE.EE.cl");
}

int runDDOT(const char* program_file){
    cl_int ciErrNum;
    int PassFailFlag = 1;

    printf("Initializing data...\n");
        PassFailFlag  = posix_memalign((void **)&h_a, 64, __nbElements * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&h_b, 64, __nbElements * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }
	h_Superacc = (bintype *) malloc(BIN_COUNT * sizeof(bintype));

	// init data
        int emax = E_BITS - log2(__nbElements);// use log in order to stay within [emin, emax]
        init_fpuniform(h_a, __nbElements, __range, emax);
        init_fpuniform(h_b, __nbElements, __range, emax);

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
	size_t size = __nbElements * sizeof(double);
	d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	d_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_b, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	d_Superacc = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, BIN_COUNT * sizeof(bintype), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    {
        printf("Initializing OpenCL DDOT...\n");
            if (__alg == 1)
                ciErrNum = initDDOT(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe);
            
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL DDOT with %u elements...\n\n", __nbElements);
            //Just a single launch or a warmup iteration
            if (__alg == 1)
                DDOT(NULL, d_Superacc, d_a, d_b, __nbElements, &ciErrNum);

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

            if (__alg == 1)
                DDOT(NULL, d_Superacc, d_a, d_b, __nbElements, &ciErrNum);

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
	double perf = 2.0 * __nbElements * sizeof(double);
	double throughput = (perf / minTime) * 1e-9;
	perf = 2.0 * __nbElements;
	perf = (perf / minTime) * 1e-9;
        printf("Alg = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", __alg, __range, __nbElements, __nbElements * sizeof(double), minTime, throughput);
        printf("Alg = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Performance = %.4f GFLOPS\n\n", __alg, __range, __nbElements, __nbElements * sizeof(double), minTime, perf);
#endif

        printf("Validating DDOT OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Superacc, CL_TRUE, 0, BIN_COUNT * sizeof(bintype), h_Superacc, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
                Superaccumulator superaccGPU((int64_t *) h_Superacc, E_BITS, F_BITS);

            printf(" ...SupersuperaccCPU()\n");
                // init accumulator
                Superaccumulator superaccCPU(E_BITS, F_BITS);
                // accumulate numbers
                for (uint i = 0; i < __nbElements; i++) {
                    superaccCPU.Accumulate(((double *) h_a)[i] * ((double *) h_b)[i]);
                }

            printf(" ...comparing the results\n");
	       superaccCPU.PrintAccumulator();
	       superaccGPU.PrintAccumulator();
               //check the results using mpfr algorithm
               /*printf("//--------------------------------------------------------\n");
	       mpfr_t *res_mpfr = sum_mpfr((double *) h_iData, __nbElements);
               PassFailFlag = superaccGPU.CompareSuperaccumulatorWithMPFR(res_mpfr);
	       double res_rounded = superaccCPU.Round();
               PassFailFlag |= superaccGPU.CompareRoundedResults(res_mpfr, res_rounded);*/

	       double roundCPU = superaccCPU.Round();
	       double roundGPU = superaccGPU.Round();
	       PassFailFlag = abs(roundCPU - roundGPU) < 1e-16 ? 1 : 0;
	       printf("[CPU] Rounded value of the compuation: %.17g\n", roundCPU);
	       printf("[GPU] Rounded value of the compuation: %.17g\n", roundGPU);
            
         //Release kernels and program
         printf("Shutting down...\n\n");
            if (__alg == 1)
                closeDDOT();
    }

    // pass or fail
    if (PassFailFlag)
	printf("[DDOT] test results...\tPASSED\n");
    else
	printf("[DDOT] test results...\tFAILED\n");

    cleanUp(EXIT_SUCCESS);
}

int runDDOTSimple(const char* program_file){
    cl_int ciErrNum;
    int PassFailFlag = 1;

    printf("Initializing data...\n");
        PassFailFlag  = posix_memalign((void **)&h_a, 64, __nbElements * sizeof(double));
        PassFailFlag |= posix_memalign((void **)&h_b, 64, __nbElements * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }

	// init data
        int emax = E_BITS - log2(__nbElements);// use log in order to stay within [emin, emax]
        init_fpuniform(h_a, __nbElements, __range, emax);
        init_fpuniform(h_b, __nbElements, __range, emax);

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
	size_t size = __nbElements * sizeof(cl_double);
	d_a = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_a, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	d_b = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, h_b, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
	d_res = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_a, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    {
        printf("Initializing OpenCL DDOT...\n");
	    if (__alg == 0)
                ciErrNum = initDDOTSimple(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL DDOT with %u elements...\n\n", __nbElements);
            //Just a single launch or a warmup iteration
            DDOTSimple(NULL, d_res, d_a, d_b, __nbElements, &ciErrNum);
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

            DDOTSimple(NULL, d_res, d_a, d_b, __nbElements, &ciErrNum);

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
	double perf = 2.0 * __nbElements * sizeof(double);
	double throughput = (perf / minTime) * 1e-9;
	perf = 2.0 * __nbElements;
	perf = (perf / minTime) * 1e-9;
        printf("Alg = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", __alg, __range, __nbElements, __nbElements * sizeof(double), minTime, throughput);
        printf("Alg = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Performance = %.4f GFLOPS\n\n", __alg, __range, __nbElements, __nbElements * sizeof(double), minTime, perf);
#endif

        printf("Validating DDOT OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_res, CL_TRUE, 0, sizeof(double), (void *)&h_res, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
		
            printf(" ...ddotCPU()\n");
                // init accumulator
		double res_cpu = DDOT_CPU(h_a, h_b, __nbElements);

            printf(" ...comparing the results\n");
		printf("[CPU]Results of simple dot product %.8g\n", res_cpu);
		printf("[GPU]Results of simple dot product %.8g\n", h_res);
	        PassFailFlag = abs(h_res - res_cpu) < 1e-16 ? 1 : 0;

         //Release kernels and program
         printf("Shutting down...\n\n");
            if (__alg == 0)
		closeDDOTSimple();
    }

    // pass or fail
    if (PassFailFlag)
	printf("[DDOT] test results...\tPASSED\n");
    else
	printf("[DDOT] test results...\tFAILED\n");

    cleanUp(EXIT_SUCCESS);
}
int cleanUp (int exitCode) {
    //Release other OpenCL Objects
    if(d_a) 
	clReleaseMemObject(d_a);
    if(d_b) 
	clReleaseMemObject(d_b);
    if(cqCommandQueue) 
	clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext) 
	clReleaseContext(cxGPUContext);

    //Release host buffers
    free(h_a);
    free(h_b);
    //free(C);
    
    return exitCode;
}

