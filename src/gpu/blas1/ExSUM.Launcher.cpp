/*
 *  Copyright (c) 2013-2015 University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "ExSUM.Launcher.hpp"
#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////
// OpenCL launcher for bitonic sort kernel
////////////////////////////////////////////////////////////////////////////////
#define EXSUM_KERNEL          "ExSUM"
#define EXSUM_COMPLETE_KERNEL "ExSUMComplete"
#define ROUND_KERNEL          "ExSUMRound"

static size_t szKernelLength;                // Byte size of kernel code
static char* cSources = NULL;                // Buffer to hold source for compilation

static cl_program       cpProgram;           //OpenCL Superaccumulator program
static cl_kernel        ckKernel;            //OpenCL Superaccumulator kernels
static cl_kernel        ckComplete;
static cl_kernel        ckRound;
static cl_command_queue cqDefaultCommandQue; //Default command queue for Superaccumulator
static cl_mem           d_Superacc;
static cl_mem           d_PartialSuperaccs;

static const uint PARTIAL_SUPERACCS_COUNT = 256; //128
static const uint WORKGROUP_SIZE          = 256;  
static const uint MERGE_WORKGROUP_SIZE    = 256; //256
static uint vector_number                 = 1;
static uint NbElements;

#ifdef AMD
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=256 -DUSE_KNUTH";
#else
static char  compileOptions[256] = "-DWARP_COUNT=16 -DWARP_SIZE=16 -DMERGE_WORKGROUP_SIZE=256 -DUSE_KNUTH -DNVIDIA -cl-mad-enable -cl-fast-relaxed-math"; // -cl-nv-verbose";
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
// GPU reduction related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initExSUM(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint NbElems,
    const uint NbFPE
){
    cl_int ciErrNum;
    NbElements = NbElems;
    vector_number = 2;

    // Read the OpenCL kernel in from source file
    FILE *program_handle;
    //printf("Load the program sources (%s)...\n", program_file);
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

    //printf("...building ExSUM program\n");
        sprintf(compileOptions, "%s -DNBFPE=%d", compileOptions, NbFPE);
        ciErrNum = clBuildProgram(cpProgram, 0, NULL, compileOptions, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            //printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);

            // Determine the reason for the error
            char buildLog[16384];
            clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), &buildLog, NULL);
            printf("%s\n", buildLog);

            return EXIT_FAILURE;
        }

    //printf("...creating ExSUM kernels:\n");
        ckKernel = clCreateKernel(cpProgram, EXSUM_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: ExSUM, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckComplete = clCreateKernel(cpProgram, EXSUM_COMPLETE_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: ExSUMComplete, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        ckRound = clCreateKernel(cpProgram, ROUND_KERNEL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateKernel: Round, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //printf("...allocating internal buffer\n");
        uint size = PARTIAL_SUPERACCS_COUNT;
        size = size * bin_count * sizeof(cl_long);
        d_PartialSuperaccs = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_PartialSuperaccs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }
        d_Superacc = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, bin_count * sizeof(bintype), NULL, &ciErrNum);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clCreateBuffer for d_Superacc, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            return EXIT_FAILURE;
        }

    //Save default command queue
    cqDefaultCommandQue = cqParamCommandQue;

    //Discard temp storage
    free(cSources);

    return EXIT_SUCCESS;
}

extern "C" void closeExSUM(void){
    cl_int ciErrNum;

    ciErrNum = clReleaseMemObject(d_PartialSuperaccs);
    ciErrNum |= clReleaseMemObject(d_Superacc);
    ciErrNum |= clReleaseKernel(ckRound);
    ciErrNum |= clReleaseKernel(ckKernel);
    ciErrNum |= clReleaseKernel(ckComplete);
    ciErrNum |= clReleaseProgram(cpProgram);
    if (ciErrNum != CL_SUCCESS) {
        printf("Error in closeExSUM(), Line %u in file %s !!!\n\n", __LINE__, __FILE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// OpenCL launchers for Superaccumulator / mergeSuperaccumulators kernels
////////////////////////////////////////////////////////////////////////////////
extern "C" size_t ExSUM(
    cl_command_queue cqCommandQueue,
    cl_mem d_Res,
    cl_mem d_Data,
    cl_int *ciErrNumRes
){
    cl_int ciErrNum;
    size_t NbThreadsPerWorkGroup, TotalNbThreads;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        NbThreadsPerWorkGroup  = WORKGROUP_SIZE;
        TotalNbThreads = PARTIAL_SUPERACCS_COUNT * NbThreadsPerWorkGroup;
        uint NbElems = NbElements / vector_number;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_mem),  (void *)&d_Data);
        ciErrNum |= clSetKernelArg(ckKernel, i++, sizeof(cl_uint), (void *)&NbElems);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    {
        NbThreadsPerWorkGroup = MERGE_WORKGROUP_SIZE;
        TotalNbThreads = NbThreadsPerWorkGroup;
        TotalNbThreads *= bin_count;

        cl_uint i = 0;
        ciErrNum = clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_Superacc);
        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_mem),  (void *)&d_PartialSuperaccs);
        ciErrNum |= clSetKernelArg(ckComplete, i++, sizeof(cl_uint), (void *)&PARTIAL_SUPERACCS_COUNT);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckComplete, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    {
        NbThreadsPerWorkGroup = MERGE_WORKGROUP_SIZE;
        TotalNbThreads = NbThreadsPerWorkGroup;

        cl_uint i = 0;
        ciErrNum  = clSetKernelArg(ckRound, i++, sizeof(cl_mem),  (void *)&d_Res);
        ciErrNum |= clSetKernelArg(ckRound, i++, sizeof(cl_mem),  (void *)&d_Superacc);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
           *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckRound, 1, NULL, &TotalNbThreads, &NbThreadsPerWorkGroup, 0, NULL, NULL);
        if (ciErrNum != CL_SUCCESS) {
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
           *ciErrNumRes = EXIT_FAILURE;
            return 0;
        }
    }

    return WORKGROUP_SIZE;
}

