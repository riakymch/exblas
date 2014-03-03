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
#define DGEMM_KERNEL "mmmKernel_local"
#define BLOCK_SIZE 16

static size_t szKernelLength;	              // Byte size of kernel code
static char* cSources = NULL;                 // Buffer to hold source for compilation

static cl_program       cpProgram;            //OpenCL Superaccumulator program
static cl_kernel        ckMatrixMul;
static cl_command_queue cqDefaultCommandQue;  //Default command queue for Superaccumulator

static const uint  VECTOR_NUMBER = 2;

#ifdef AMD
static char  compileOptions[256] = "";
#else
static char  compileOptions[256] = "-DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
#endif


extern "C" cl_int initDGEMMAMD(
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

    printf("...building program\n");
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

    printf("...creating DGEMM kernel:\n");
        ckMatrixMul = clCreateKernel(cpProgram, DGEMM_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeDGEMMAMD(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseKernel(ckMatrixMul);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeDGEMM(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Superaccumulator / mergeSuperaccumulators kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t DGEMMAMD(
    cl_command_queue cqCommandQueue,
    Matrix d_C,
    const Matrix d_A,
    const Matrix d_B,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        size_t NbThreadsPerWorkGroup[] = {BLOCK_SIZE, BLOCK_SIZE};
	size_t widthB = d_B.width / VECTOR_NUMBER;
	size_t heightA = d_A.height / VECTOR_NUMBER;
	size_t TotalNbThreads[] = {widthB, heightA};
	size_t neededLocalMemory = (BLOCK_SIZE * VECTOR_NUMBER) * (BLOCK_SIZE * VECTOR_NUMBER) * sizeof(cl_double);

	cl_int i = 0;
        ciErrNum  = clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_C.elements);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_A.elements);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_B.elements);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_int),  (void *)&d_A.width);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, neededLocalMemory,  NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckMatrixMul, 2, NULL, TotalNbThreads, NbThreadsPerWorkGroup, 0, 0, 0);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	    *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }
    return EXIT_SUCCESS;
}
