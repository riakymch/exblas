
#ifndef TRSV_HPP_INCLUDED
#define TRSV_HPP_INCLUDED

#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <gmp.h>
#include <mpfr.h>

extern "C" int TRSVUNN(
    double *x,
    const double *a,
    const int n
);

extern "C" int TRSVLNU(
    double *x,
    const double *a,
    const int n
);

extern "C" bool compare(
    const double *trsv_cpu,
    const double *trsv_gpu,
    const uint n,
    const double epsilon
);

extern "C" bool compareTRSVUNNToMPFR(
    const double *a,
    double *b,
    const double *trsv,
    const int n,
    const double epsilon
);

extern "C" bool compareTRSVLNUToMPFR(
    const double *a,
    double *b,
    const double *trsv,
    const int n,
    const double epsilon
);

extern "C" bool verifyTRSVUNN(
    const double *a,
    const double *b,
    const double *x,
    const int n,
    const double epsilon
);

extern "C" bool verifyTRSVLNU(
    const double *a,
    const double *b,
    const double *x,
    const int n,
    const double epsilon
);

extern "C" void printMatrix(
    const int iscolumnwise,
    const double *A,
    const uint lda,
    const uint n
);

extern "C" void printVector(
    const double *a,
    const uint n
);

////////////////////////////////////////////////////////////////////////////////
// GPU TRSV product related functions
////////////////////////////////////////////////////////////////////////////////
extern "C" cl_int initTRSV(
    cl_context cxGPUContext,
    cl_command_queue cqParamCommandQue,
    cl_device_id cdDevice,
    const char* program_file,
    const uint n,
    const uint alg,
    const uint NbFPE
);

extern "C" void closeTRSV(
    void
);

extern "C" size_t TRSV(
    cl_command_queue cqCommandQueue,
    cl_mem d_x,
    const cl_mem d_a,
    const cl_mem d_b,
    const uint n,
    cl_int *ciErrNum
);

#endif
