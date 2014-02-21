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
cl_mem            d_iData, d_oData, d_Histogram;      //OpenCL memory buffer objects
void              *h_iData;
double            h_oData;
bintype           *h_HistogramGPU;

static uint __nbElements = 0;
static uint __range      = 0;
static uint __nbfpe      = 0;
static uint __alg        = 0;

static void __usage(int argc __attribute__((unused)), char **argv) {
  fprintf(stderr, "Usage: %s [-n number of elements  -r range -e nbfpe -a alg (0-acc, 1-fpe, 2-fpeee, 3-red, 4-dem)] \n", argv[0]);
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

  if (__nbElements == 0) {
    __usage(argc, argv);
    exit(-1);
  }
  if (__alg > 4) {
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
    printf("Starting with an array of %i double elements\n\n",  __nbElements); 

    if (__alg == 0) 
        runSuperaccumulator("../src/Superaccumulator.cl");
    else if (__alg == 1)
        runSuperaccumulator("../src/Superaccumulator.FPE.cl");
    else if (__alg == 2)
        runSuperaccumulator("../src/Superaccumulator.FPE.EX.cl");
    else if (__alg == 3)
        runReduction("../src/Reduction.cl");
    else if (__alg == 4)
        runSuperaccumulator("../src/Superaccumulator.Demmel.cl");
}

int runSuperaccumulator(const char* program_file){
    cl_int ciErrNum;
    int    PassFailFlag = 1;

    printf("Initializing data...\n");
        //h_iData         = (void    *) malloc(__nbElements * sizeof(double));
        PassFailFlag = posix_memalign(&h_iData, 64, __nbElements * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }
        h_HistogramGPU = (bintype *) malloc(BIN_COUNT  * sizeof(bintype));
	// init data
        int emax = E_BITS - log2(__nbElements);
        init_fpuniform((double *) h_iData, __nbElements, __range, emax); // 2000

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
        d_iData = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, __nbElements * sizeof(cl_double), h_iData, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_iData, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        d_Histogram = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, BIN_COUNT * sizeof(bintype), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_Histogram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    
    {
        printf("Initializing OpenCL superaccumulator...\n");
            ciErrNum = initSuperaccumulator(cxGPUContext, cqCommandQueue, cdDevice, program_file, __nbfpe);
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL superaccumulator with %u elements...\n\n", __nbElements);
            //Just a single launch or a warmup iteration
            Superaccumulate(NULL, d_Histogram, d_iData, __nbElements, &ciErrNum);
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

            Superaccumulate(NULL, d_Histogram, d_iData, __nbElements, &ciErrNum);

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
            gpuTime[iter] = 1e-9 * ((unsigned long)endTime - (unsigned long)startTime); // / (double)NUM_ITER;
        }
    
	double minTime = min(gpuTime, NUM_ITER);
        //printf("\nOpenCL time: %.5f s\n\n", 1.0e-9 * ((double)endTime - (double)startTime)/(double)NUM_ITER);
        printf("Alg = %u \t NbFPE = %u \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", 
          __alg, __nbfpe, __range, __nbElements, __nbElements * sizeof(double), minTime, ((1e-9 * __nbElements * sizeof(double)) / minTime));
#endif

        printf("Validating OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Histogram, CL_TRUE, 0, BIN_COUNT * sizeof(bintype), h_HistogramGPU, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
                Superaccumulator accumulatorGPU((int64_t *) h_HistogramGPU, E_BITS, F_BITS);

            printf(" ...SuperaccumulatorCPU()\n");
                // init accumulator
                Superaccumulator accumulatorCPU(E_BITS, F_BITS);
                // accumulate numbers
                for (uint i = 0; i < __nbElements; i++) {
                    accumulatorCPU.Accumulate(((double *) h_iData)[i]);
                }

            printf(" ...comparing the results\n");
	       //accumulatorCPU.PrintAccumulator();
	       //accumulatorGPU.PrintAccumulator();
               //check the results using mpfr algorithm
               printf("//--------------------------------------------------------\n");
	       char *res_mpfr = sum_mpfr((double *) h_iData, __nbElements);
               //accumulatorCPU.CompareSuperaccumulatorWithMPFR(res_mpfr);
               accumulatorGPU.CompareSuperaccumulatorWithMPFR(res_mpfr);
            
  	       //print the final result of using superaccumulator
               //printf("//--------------------------------------------------------\n");
               //roundSuperaccumulator(h_HistogramGPU);

  	       //check the results using the Kahan summation
               //printf("//--------------------------------------------------------\n");
               //roundKahan((double *) h_iData, __nbElements);
               //printf("//--------------------------------------------------------\n");

            //Release kernels and program
         printf("Shutting down...\n\n");
            closeSuperaccumulator();
    }

    // pass or fail
    if (!PassFailFlag)
	printf("[SuperaccumulatorFPE] test results...\nPASSED\n");
    else
	printf("[SuperaccumulatorFPE] test results...\nFAILED\n");

    cleanUp(EXIT_SUCCESS);
}

int runReduction(const char* program_file){
    cl_int ciErrNum;
    int    PassFailFlag = 1;

    printf("Initializing data...\n");
        PassFailFlag = posix_memalign(&h_iData, 64, __nbElements * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }
	// init data
        int emax = E_BITS - log2(__nbElements);
        init_fpuniform((double *) h_iData, __nbElements, __range, emax); // 2000

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
        d_iData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, __nbElements * sizeof(cl_double), h_iData, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_iData, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        d_oData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_iData, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    {
        printf("Initializing OpenCL Reduction...\n");
            ciErrNum = initReduction(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL Reduction with %u elements...\n\n", __nbElements);
            //Just a single launch or a warmup iteration
            Reduction(NULL, d_oData, d_iData, __nbElements, &ciErrNum);
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

            Reduction(NULL, d_oData, d_iData, __nbElements, &ciErrNum);

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
        printf("Alg = 2 \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", 
            __range, __nbElements, __nbElements * sizeof(double), minTime, ((1e-9 * __nbElements * sizeof(double)) / minTime));
#endif

        printf("Validating Reduction OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_oData, CL_TRUE, 0, sizeof(double), &h_oData, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
		printf("\nGPU Parallel Reduction: %.8g\n\n", h_oData);
            //Release kernels and program
         printf("Shutting down...\n\n");
            closeReduction();
    }

    // pass or fail
    if (!PassFailFlag)
	printf("[Reduction] test results...\nPASSED\n");
    else
	printf("[Reduction] test results...\nFAILED\n");

    cleanUp(EXIT_SUCCESS);
}

/*
int runReductionOpenCLInAction(const char* program_file){
    cl_int ciErrNum;
    int    PassFailFlag = 1;

    printf("Initializing data...\n");
        //h_iData         = (void    *) malloc(__nbElements * sizeof(double));
        PassFailFlag = posix_memalign(&h_iData, 64, __nbElements * sizeof(double));
        if (PassFailFlag != 0) {
            printf("ERROR: could not allocate memory with posix_memalign!\n");
            exit(1);
        }
	// init data
        int emax = E_BITS - log2(__nbElements);
        init_fpuniform((double *) h_iData, __nbElements, __range, emax); // 2000

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
        d_iData = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, __nbElements * sizeof(cl_double), h_iData, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_iData, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
        d_oData = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_double), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_iData, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            cleanUp(EXIT_FAILURE);
        }
    {
        printf("Initializing OpenCL Reduction...\n");
            ciErrNum = initReduction(cxGPUContext, cqCommandQueue, cdDevice, program_file);
            if (ciErrNum != CL_SUCCESS)
                cleanUp(EXIT_FAILURE);

        printf("Running OpenCL Reduction with %u elements...\n\n", __nbElements);
            //Just a single launch or a warmup iteration
            Reduction(NULL, d_iData, d_oData, __nbElements, &ciErrNum);
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

            Reduction(NULL, d_iData, d_oData, __nbElements, &ciErrNum);

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
            gpuTime[iter] = 1.0e-9 * ((double)endTime - (double)startTime); // / (double)NUM_ITER;
        }

	double minTime = min(gpuTime, NUM_ITER);
        printf("Alg = 2 \t Range = %u \t NbElements = %u \t Size = %lu \t Time = %.8f s \t Throughput = %.4f GB/s\n\n", 
            __range, __nbElements, __nbElements * sizeof(double), minTime, ((1e-9 * __nbElements * sizeof(double)) / minTime));
#endif

        printf("Validating Reduction OpenCL results...\n");
            printf(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_oData, CL_TRUE, 0, sizeof(double), &h_oData, 0, NULL, NULL);
                if (ciErrNum != CL_SUCCESS) {
                    printf("Error in clEnqueueReadBuffer Line %u in file %s !!!\n\n", __LINE__, __FILE__);
                    cleanUp(EXIT_FAILURE);
                }
		printf("\nGPU Parallel Reduction: %.8g\n\n", h_oData);

            //Release kernels and program
         printf("Shutting down...\n\n");
            closeReduction();
    }

    // pass or fail
    if (!PassFailFlag)
	printf("[Reduction] test results...\nPASSED\n");
    else
	printf("[Reduction] test results...\nFAILED\n");

    cleanUp(EXIT_SUCCESS);
}
*/

int cleanUp (int exitCode) {
    //Release other OpenCL Objects
    if(d_iData) 
	clReleaseMemObject(d_iData);
    if(d_Histogram)
	clReleaseMemObject(d_Histogram);
    if(cqCommandQueue) 
	clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext) 
	clReleaseContext(cxGPUContext);

    //Release host buffers
    free(h_iData); 
    free(h_HistogramGPU);
    
    return exitCode;
}

