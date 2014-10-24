
#ifndef TRSV_HPP_INCLUDED
#define TRSV_HPP_INCLUDED

#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <gmp.h>
#include <mpfr.h>

extern "C" int TRSVUNN(
    const double *a,
    const double *b,
    const int n,
    double *x
);

extern "C" int TRSVUNN_Kulisch(
    const double *a,
    const double *b,
    const int N,
    double *x
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

extern "C" double verifyTRSVUNN(
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

extern "C" double condA(
    const double *a,
    const int n
);

extern "C" double TwoProductFMA(
    double a,
    double b,
    double *d
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
