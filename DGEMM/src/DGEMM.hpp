
#ifndef DGEMM_HPP_INCLUDED
#define DGEMM_HPP_INCLUDED

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include <gmp.h>
#include <mpfr.h>

void DGEMMCPU(
    double *C,
    const double *A,
    const double *B,
    const uint y,
    const uint x,
    const uint z
);

extern "C" bool compare(
    const double *ref_dgemm,
    const double *dgemm,
    const uint length,
    const double epsilon
);

extern "C" bool compareDGEMMWithMPFR(
    const double *dgemm,
    const double *h_a,
    const double *h_b,
    const uint m,
    const uint n,
    const uint k,
    const double epsilon
);

void printMatrix(
    const double *A,
    const uint m,
    const uint n
);

#endif
