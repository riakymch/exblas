
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string.h>
#include <cfloat>
#include "common.hpp"

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
        err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_VERSION, 128 * sizeof(char), pPlatformName, NULL);
        printf("platform version : %s\n", pPlatformName);	
        err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
        if ( err != CL_SUCCESS ) {
            printf("ERROR: Failed to retreive platform vendor name.\n");
            return NULL;
        }

        printf("### Platform[%i] : %s\n", ui, pPlatformName);

        if (!strcmp(pPlatformName, name))
            ui_res = ui; //return pPlatforms[ui];
    }
    printf("### Using Platform : %s\n", name);

    if (ui_res > -1)
        return pPlatforms[ui_res];
    else
        return NULL;
}

cl_device_id GetOCLDevice(cl_platform_id pPlatform, char name[]) {
    printf("clGetDeviceIDs...\n");

    cl_device_id dDevices[10] = { 0 };
    char dDeviceName[128] = { 0 };

    cl_uint uiNumDevices = 0;
    cl_int err = clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_GPU, 10, dDevices, &uiNumDevices);
    cl_int uiRes = -1;

    for (cl_int ui = 0; ui < (cl_int) uiNumDevices; ++ui) {
        //cl_int compute_units;
        //err = clGetDeviceInfo(dDevices[ui], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &compute_units, NULL);
        //printf("compute_units = %d\n", compute_units);
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

cl_device_id GetOCLDevice(cl_platform_id pPlatform) {
    printf("clGetDeviceIDs...\n"); 

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

        printf("### Device[%i] : %s\n", ui, dDeviceName);
        if (ui == 0)
            strcpy(name, dDeviceName);
    }
    printf("### Using Device : %s\n", name);

    return dDevices[0];
}

inline double randDouble(int emin, int emax, int neg_ratio) {
    // Uniform mantissa
    double x = double(rand()) / double(RAND_MAX * .99) + 1.;
    // Uniform exponent
    int e = (rand() % (emax - emin)) + emin;
    // Sign
    if(neg_ratio > 1 && rand() % neg_ratio == 0) {
        x = -x;
    }
    return ldexp(x, e);
}

double min(double arr[], int size) {
    assert(arr != NULL);
    assert(size >= 0);

    if ((arr == NULL) || (size <= 0))
       return NAN;

    double val = DBL_MAX; 
    for (int i = 0; i < size; i++)
        if (val > arr[i])
            val = arr[i];

    return val;
}

void init_fpuniform(double *array, int size, int range, int emax) {
    /*//Generate numbers on several bins starting from emax
    for(int i = 0; i != size; ++i)
        //array[i] = randDouble(emax-range, emax, 1);
        array[i] = randDouble(0, range, 1);*/

    //Generate numbers on an interval [1, 2]
    for(int i = 0; i != size; ++i)
        array[i] = 1.0 + double(rand()) / double(RAND_MAX);
}

void print2Superaccumulators(bintype *binCPU, bintype *binGPU) {
    uint i;
    for (i = 0; i < BIN_COUNT; i++)
        printf("bin[%3d]: %lX \t %lX\n", i, binCPU[i], binGPU[i]);
}


////////////////////////////////////////////////////////////////////////////////
// Reference CPU superaccumulator
////////////////////////////////////////////////////////////////////////////////
extern "C" double roundSuperaccumulator(
    bintype *bin
);


