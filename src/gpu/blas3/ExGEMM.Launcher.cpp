/*
 *  Copyright (c) 2013-2015 University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "ExGEMM.Launcher.hpp"
#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define DGEMM_KERNEL "matrixMul"
#ifdef AMD
  #define BLOCK_SIZE 16
#else
  #define BLOCK_SIZE 32
#endif

static size_t szKernelLength;                 // Byte size of kernel code
static char* cSources = NULL;                 // Buffer to hold source for compilation

static cl_program       cpProgram;            //OpenCL Superaccumulator program
static cl_kernel        ckMatrixMul;
static cl_command_queue cqDefaultCommandQue;  //Default command queue for Superaccumulator

#ifdef AMD
static char  compileOptions[256] = "-DBLOCK_SIZE=16 -DUSE_KNUTH";
#else
static char  compileOptions[256] = "-DBLOCK_SIZE=32 -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math";
#endif


////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
cl_platform_id GetOCLPlatform(char name[]) {
  cl_platform_id pPlatforms[10] = { 0 };
  char pPlatformName[128] = { 0 };

  cl_uint uiPlatformsCount = 0;
  cl_int err = clGetPlatformIDs(10, pPlatforms, &uiPlatformsCount);
  cl_int ui_res = -1;

  for (cl_int ui = 0; ui < (cl_int) uiPlatformsCount; ++ui) {
      err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
      if ( err != CL_SUCCESS ) {
        printf("ERROR: Failed to retreive platform vendor name.\n");
        return NULL;
      }

      //printf("### Platform[%i] : %s\n", ui, pPlatformName);

      if (!strcmp(pPlatformName, name))
        ui_res = ui; //return pPlatforms[ui];
  }
  //printf("### Using Platform : %s\n", name);

  if (ui_res > -1)
    return pPlatforms[ui_res];
  else
    return NULL;
}

cl_device_id GetOCLDevice(cl_platform_id pPlatform) {
  cl_device_id dDevices[10] = { 0 };
  char name[128] = { 0 };
  char dDeviceName[128] = { 0 };

  cl_uint uiNumDevices = 0;
  cl_int err = clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_GPU, 10, dDevices, &uiNumDevices);

  for (cl_int ui = 0; ui < (cl_int) uiNumDevices; ++ui) {
      err = clGetDeviceInfo(dDevices[ui], CL_DEVICE_NAME, 128 * sizeof(char), dDeviceName, NULL);
      if ( err != CL_SUCCESS ) {
        printf("ERROR: Failed to retreive platform vendor name.\n");
        return NULL;
      }

      //printf("### Device[%i] : %s\n", ui, dDeviceName);
      if (ui == 0)
        strcpy(name, dDeviceName);
  }
  //printf("### Using Device : %s\n", name);

  return dDevices[0];
}

cl_device_id GetOCLDevice(cl_platform_id pPlatform, char name[]) {
  cl_device_id dDevices[10] = { 0 };
  char dDeviceName[128] = { 0 };

  cl_uint uiNumDevices = 0;
  cl_int err = clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_GPU, 10, dDevices, &uiNumDevices);
  cl_int uiRes = -1;

  for (cl_int ui = 0; ui < (cl_int) uiNumDevices; ++ui) {
    err = clGetDeviceInfo(dDevices[ui], CL_DEVICE_NAME, 128 * sizeof(char), dDeviceName, NULL);
    if ( err != CL_SUCCESS ) {
      printf("ERROR: Failed to retreive platform vendor name.\n");
      return NULL;
    }

    printf("### Device[%i] : %s\n", ui, dDeviceName);

    if (!strcmp(dDeviceName, name))
      uiRes = ui;
  }
  printf("### Using Device : %s\n", name);

  if (uiRes > -1)
    return dDevices[uiRes];
  else
    return NULL;
}


////////////////////////////////////////////////////////////////////////////////
// GPU related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExGEMM(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbFPE
){
    cl_int ciErrNum;

    // Read the OpenCL kernel in from source file
    FILE *program_handle;
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

    //printf("clCreateProgramWithSource...\n"); 
        cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSources, &szKernelLength, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //printf("...building program\n");
        sprintf(compileOptions, "%s -DNBFPE=%d", compileOptions, NbFPE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        //if (ciErrNum != CL_SUCCESS) {
            //printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[4096]; // 16384
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            //return EXIT_FAILURE;
        //}

    //printf("...creating DGEMM kernel:\n");
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

extern "C" void closeExGEMM(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseKernel(ckMatrixMul);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeDGEMM(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for the kernel
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t runExDGEMM(
    cl_command_queue cqCommandQueue,
    cl_mem d_C,
    const cl_mem d_A,
    const cl_mem d_B,
    const uint m,
    const uint n,
    const int multi,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        size_t NbThreadsPerWorkGroup[] = {(size_t) BLOCK_SIZE, (size_t) BLOCK_SIZE};
        size_t TotalNbThreads[] = {(size_t) (n), (size_t) (m / multi)};
        size_t neededLocalMemory = BLOCK_SIZE * BLOCK_SIZE * sizeof(cl_double);

        cl_int i = 0;
        ciErrNum  = clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_C);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_A);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_mem),  (void *)&d_B);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_int),  (void *)&m);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, sizeof(cl_int),  (void *)&n);
        ciErrNum |= clSetKernelArg(ckMatrixMul, i++, neededLocalMemory,  NULL);
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

