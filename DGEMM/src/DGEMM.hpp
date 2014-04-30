
#ifndef DGEMM_HPP_INCLUDED
#define DGEMM_HPP_INCLUDED

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

void DGEMMCPU(
    double *C,
    const double *A,
    const double *B,
    const uint y,
    const uint x,
    const uint z
);


int compare(
    const double *ref_dgemm,
    const double *dgemm,
    const uint length,
    const double epsilon
);

void printMatrix(
    const double *A,
    const uint m,
    const uint n
);
#endif
