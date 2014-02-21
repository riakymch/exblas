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

#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define SUPERACCUMULATOR_KERNEL        "Superaccumulator"
#define MERGE_SUPERACCUMULATORS_KERNEL "mergeSuperaccumulators"

static size_t szKernelLength;	              // Byte size of kernel code
static char* cSources = NULL;                 // Buffer to hold source for compilation

static cl_program       cpProgram;            //OpenCL Superaccumulator program
static cl_kernel        ckSuperacc;
static cl_kernel        ckMergeSuperaccs;     //OpenCL Superaccumulator kernels
static cl_command_queue cqDefaultCommandQue;  //Default command queue for Superaccumulator
static cl_mem           d_PartialAccumulators;
//static cl_mem           d_Overflow;

//static const uint  PARTIAL_ACCUMULATORS_COUNT = 2048;
static const uint  PARTIAL_ACCUMULATORS_COUNT = 1;
static const uint  WARP_COUNT                 = 16;
static const uint  WARP_SIZE                  = 16;
static const uint  MERGE_WORKGROUP_SIZE       = 256;
static const uint  VECTOR_NUMBER              = 2;

#ifdef AMD
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=256 -DUSE_KNUTH";
#else
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=256 -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
//static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=256 -DNVIDIA -cl-nv-verbose";
#endif


extern "C" cl_int initDGEMM(
    cl_context cxGPUContext, 
    cl_command_queue cqParamCommandQue, 
    cl_device_id cdDevice,
    const char* program_file
){
    cl_int ciErrNum;
    size_t kernelLength;

    // Read the OpenCL kernel in from source file
    FILE *program_handle;
    printf("Load the program sources (%s)...\n", program_file);
    program_handle = fopen(program_file, "r");
    if (!program_handle) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    szKernelLength = ftell(program_handle);
    rewind(program_handle);
    cSources = (char *) malloc(szKernelLength + 1);
    cSources[szKernelLength] = '\0';
    ciErrNum = fread(cSources, sizeof(char), szKernelLength, program_handle);
    fclose(program_handle);

    printf("clCreateProgramWithSource...\n"); 
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSources, &kernelLength, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    printf("...building Superaccumulator program\n");
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            //printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[4096]; // 16384
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }
	
    /*//Get the binary
    size_t nb_devices, nbread;
    ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &nb_devices, &nbread);// Return 1 devices
    printf("nb_devices = %d\n", nb_devices);
    size_t *np = new size_t[nb_devices];//Create size array
    ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*nb_devices, np, &nbread);//Load in np the size of my binary  
    char** bn = new char* [nb_devices]; //Create the binary array   
    for(int i =0; i < nb_devices;i++)
        bn[i] = new char[np[i]]; // I know... it's bad... but if i use new char[np[i]], i have a segfault... :/  
    ciErrNum = clGetProgramInfo(cpProgram, CL_PROGRAM_BINARIES, sizeof(unsigned char *)*nb_devices, bn, &nbread); //Load the binary itself    
    //printf("%s\n", bn[0]); //Print the first binary. But here, I have some curious characters  
    FILE *fp = fopen("Superaccumulator.cl.bin", "wb");  
    fwrite(bn[0], sizeof(bn[0]), np[0], fp); // Save the binary, but my file stay empty  
    fclose(fp);*/

    printf("...creating Superaccumulator kernels:\n");
        ckSuperacc = clCreateKernel(cpProgram, SUPERACCUMULATOR_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: Superaccumulator, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckMergeSuperaccs = clCreateKernel(cpProgram, MERGE_SUPERACCUMULATORS_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: mergeSuperaccumulators, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
    printf("...allocating internal buffer\n");
        d_PartialAccumulators = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, PARTIAL_ACCUMULATORS_COUNT * BIN_COUNT * sizeof(cl_long), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        //d_Overflow = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &ciErrNum);

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeDGEMM(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseMemObject(d_PartialAccumulators);
    ciErrNum |= clReleaseKernel(ckSuperacc);
    ciErrNum |= clReleaseKernel(ckMergeSuperaccs);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeSuperaccumulator(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Superaccumulator / mergeSuperaccumulators kernels
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b){
    return a - a % b;
}

extern "C" size_t DGEMM(
    cl_command_queue cqCommandQueue,
    cl_mem d_Accumulator,
    cl_mem d_Data,
    uint NbElements,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;
    size_t NbThreadsPerWorkGroup, TotalNbThreads;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        NbThreadsPerWorkGroup  = WARP_SIZE * WARP_COUNT;
        TotalNbThreads = PARTIAL_ACCUMULATORS_COUNT * NbThreadsPerWorkGroup;
        NbElements = NbElements / VECTOR_NUMBER;

        ciErrNum  = clSetKernelArg(ckSuperacc, 0, sizeof(cl_mem),  (void *)&d_PartialAccumulators);
        ciErrNum |= clSetKernelArg(ckSuperacc, 1, sizeof(cl_mem),  (void *)&d_Data);
        //ciErrNum |= clSetKernelArg(ckSuperacc, 2, sizeof(cl_mem), (void *)&d_Overflow);
        ciErrNum |= clSetKernelArg(ckSuperacc, 2, sizeof(cl_uint), (void *)&NbElements);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckSuperacc, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
	/*int *h_Overflow = (int *) malloc(sizeof(int));
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Overflow, CL_TRUE, 0, sizeof(cl_int), h_Overflow, 0, NULL, NULL);
	printf("Overflows in Superaccumulator() = %d \t", *h_Overflow);*/
    }
    {
        NbThreadsPerWorkGroup = MERGE_WORKGROUP_SIZE;
        //TotalNbThreads = iDivUp(PARTIAL_ACCUMULATORS_COUNT, NbThreadsPerWorkGroup) * NbThreadsPerWorkGroup;
        TotalNbThreads = BIN_COUNT * NbThreadsPerWorkGroup;

        ciErrNum  = clSetKernelArg(ckMergeSuperaccs, 0, sizeof(cl_mem),  (void *)&d_Accumulator);
        ciErrNum |= clSetKernelArg(ckMergeSuperaccs, 1, sizeof(cl_mem),  (void *)&d_PartialAccumulators);
        //ciErrNum |= clSetKernelArg(ckMergeSuperaccs, 2, sizeof(cl_mem), (void *)&d_Overflow);
        ciErrNum |= clSetKernelArg(ckMergeSuperaccs, 2, sizeof(cl_uint), (void *)&PARTIAL_ACCUMULATORS_COUNT);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckMergeSuperaccs, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
	/*int *h_Overflow = (int *) malloc(sizeof(int));
        ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Overflow, CL_TRUE, 0, sizeof(cl_int), h_Overflow, 0, NULL, NULL);
	printf(" Merge() = %d\n", *h_Overflow);*/
    }

    return (WARP_SIZE * WARP_COUNT);
}
