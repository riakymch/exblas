
#ifndef TRSV_HPP_INCLUDED
#define TRSV_HPP_INCLUDED

#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <gmp.h>
#include <mpfr.h>

double TRSVCPU(
    double *a,
    double *b,
    const unsigned int n
);

extern "C" mpfr_t *TRSVWithMPFR(
    double *h_a,
    double *h_b,
    int size);

extern "C" bool CompareWithMPFR(
    mpfr_t *res_mpfr,
    double res_rounded
);

#endif
