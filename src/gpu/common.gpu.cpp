/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "common.gpu.hpp"


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

#if 1
cl_device_id GetOCLDevice(cl_platform_id pPlatform) {
  cl_device_id dDevices[10] = { 0 };
  char name[128] = { 0 };
  char dDeviceName[128] = { 0 };

  cl_uint uiNumDevices = 0;
  cl_int err = clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_GPU, 10, dDevices, &uiNumDevices);
  if (err != CL_SUCCESS) {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
	exit(0);
  }

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
#else
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
#endif

