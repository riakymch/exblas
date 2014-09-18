
#ifndef DDOT_HPP_INCLUDED
#define DDOT_HPP_INCLUDED

#include <ostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <gmp.h>
#include <mpfr.h>

double DDOTCPU(
    double *a,
    double *b,
    const unsigned int n
);

extern "C" mpfr_t *DDOTWithMPFR(
    double *h_a,
    double *h_b,
    int size);

extern "C" bool CompareWithMPFR(
    mpfr_t *res_mpfr,
    double res_rounded
);

#endif
