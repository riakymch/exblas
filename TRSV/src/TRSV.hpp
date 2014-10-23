
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

extern "C" void printMatrix(
    const double *A,
    const uint n
);

extern "C" void printVector(
    const double *a,
    const uint n
);

#endif
