
#ifndef DGEMM_HPP_INCLUDED
#define DGEMM_HPP_INCLUDED

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

void matrixMultiplicationCPUReference(
    cl_double *output,
    cl_double *input0,
    cl_double *input1,
    const cl_uint y,
    const cl_uint x,
    const cl_uint z
);


int compare(
    const double *refData,
    const double *data,
    const int length,
    const double epsilon
);

void printMatrix(
    double *A,
    int m,
    int n
);
#endif
