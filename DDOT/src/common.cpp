
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string.h>
#include <cfloat>
#include "common.hpp"

////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
cl_platform_id GetOCLPlatform(char name[])
{
  cl_platform_id pPlatforms[10] = { 0 };
  char pPlatformName[128] = { 0 };

  cl_uint uiPlatformsCount = 0;
  cl_int err = clGetPlatformIDs(10, pPlatforms, &uiPlatformsCount);
  cl_int ui_res = -1;

  for (cl_int ui = 0; ui < (cl_int) uiPlatformsCount; ++ui)
    {
      err = clGetPlatformInfo(pPlatforms[ui], CL_PLATFORM_NAME, 128 * sizeof(char), pPlatformName, NULL);
      if ( err != CL_SUCCESS )
	{
	  printf("ERROR: Failed to retreive platform vendor name.\n");
	  return NULL;
        }

      printf("### Platform[%i] : %s\n", ui, pPlatformName);

      if (!strcmp(pPlatformName, name))
	ui_res = ui; //return pPlatforms[ui];
    }
  printf("### Using Platform : %s\n", name);

  if (ui_res > -1){
    return pPlatforms[ui_res];
  }
  else {
    return NULL;
  }
}

cl_device_id GetOCLDevice(cl_platform_id pPlatform)
{
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

cl_device_id GetOCLDevice(cl_platform_id pPlatform, char name[])
{
  printf("clGetDeviceIDs...\n"); 

  cl_device_id dDevices[10] = { 0 };
  char dDeviceName[128] = { 0 };

  cl_uint uiNumDevices = 0;
  cl_int err = clGetDeviceIDs(pPlatform, CL_DEVICE_TYPE_GPU, 10, dDevices, &uiNumDevices);
  cl_int uiRes = -1;

  for (cl_int ui = 0; ui < (cl_int) uiNumDevices; ++ui)
    {
      err = clGetDeviceInfo(dDevices[ui], CL_DEVICE_NAME, 128 * sizeof(char), dDeviceName, NULL);
      if ( err != CL_SUCCESS )
	{
	  printf("ERROR: Failed to retreive platform vendor name.\n");
	  return NULL;
        }

      printf("### Device[%i] : %s\n", ui, dDeviceName);

      if (!strcmp(dDeviceName, name))
	uiRes = ui;
    }
  printf("### Using Device : %s\n", name);

  if (uiRes > -1){
    return dDevices[uiRes];
  }
  else {
    return NULL;
  }
}

inline double randDouble(int emin, int emax, int neg_ratio)
{
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

void init_fpuniform(double *array, int size, int range, int emax)
{
    /*//Generate numbers on several bins starting from emax
    for(int i = 0; i != size; ++i) {
        //array[i] = randDouble(emax-range, emax, 1);
        array[i] = randDouble(0, range, 1);
    }*/
    //Generate numbers on an interval [1, 2]
    for(int i = 0; i != size; ++i) {
        array[i] = 1.0 + double(rand()) / double(RAND_MAX);
    }
}

void print2Superaccumulators(bintype *binCPU, bintype *binGPU) {
  uint i;
  for (i = 0; i < BIN_COUNT; i++)
    printf("bin[%3d]: %llX \t %llX\n", i, binCPU[i], binGPU[i]);
}


////////////////////////////////////////////////////////////////////////////////
// Reference CPU superaccumulator
////////////////////////////////////////////////////////////////////////////////
extern "C" double roundSuperaccumulator(
    bintype *bin
);


////////////////////////////////////////////////////////////////////////////////
// MPFR and Kahan summation functions
////////////////////////////////////////////////////////////////////////////////
extern "C" mpfr_t *ddotWithMPFR(double *h_a, double *h_b, int size) {
  mpfr_t *sum, ddot, op1;
  int i;
  sum = (mpfr_t *) malloc(sizeof(mpfr_t));

  mpfr_init2(op1, 64);
  mpfr_init2(ddot, 128);
  mpfr_init2(*sum, 2098);

  mpfr_set_d(ddot, 0.0, MPFR_RNDN);
  mpfr_set_d(*sum, 0.0, MPFR_RNDN);

  for (i = 0; i < size; i++) {
    mpfr_set_d(op1, h_a[i], MPFR_RNDN);
    mpfr_mul_d(ddot, op1, h_b[i], MPFR_RNDN);
    mpfr_add(*sum, *sum, ddot, MPFR_RNDN);
  }

  mpfr_free_cache();

  return sum;
}

extern "C" bool CompareWithMPFR(mpfr_t *res_mpfr, double res_rounded) {
  double rounded_mpfr = mpfr_get_d(*res_mpfr, MPFR_RNDD);
  printf("GPU Parallel DDOT: %.17g\n", res_rounded);
  printf("Rounded value of MPFR: %.17g\n", rounded_mpfr);

  //Compare the results with MPFR using native functions
  bool res_cmp = false;
  if (rounded_mpfr == res_rounded){
      printf("\t The result is EXACT -- matches the MPFR algorithm!\n\n");
      res_cmp = true;
  } else {
      printf("\t The result is WRONG -- does not match the MPFR algorithm!\n\n");
  }

  mpfr_clear(*res_mpfr);
  free(res_mpfr);
  mpfr_free_cache();

  return res_cmp;
}

extern "C" double KnuthTwoSum(double a, double b, double *s) {
    double r = a + b;
    double z = r - a;
    *s = (a - (r - z)) + (b - z);
    return r;
}

extern "C" double TwoProductFMA(double a, double b, double *d) {
    double p = a * b;
    *d = fma(a, b, -p);
    return p;
}

/*
 * Kahan Summation :
 * We use Kahan summation for an accurate sum of large arrays.
 * http://en.wikipedia.org/wiki/Kahan_summation_algorithm
 */
inline void KahanSummation(double *s, double *c, double d) {
  double y, t;

  y = d - *c;
  t = *s + y;
  *c = (t - *s) - y;
  *s = t;
}

extern "C" double roundKahan(double *data, int size) {
  double r1, r2;
  int i;

  r1 = 0.;
  r2 = 0.;
  for (i = 0; i < size; i++)
    KahanSummation(&r1, &r2, data[i]);

  //printf("\tKahan Summation  : %a %a \n", r1, r2);
  printf("\tKahan Result     : %.52g \n", r1);
  printf("\tKahan Error      : %.52g \n", r2);

  return r1;
}
