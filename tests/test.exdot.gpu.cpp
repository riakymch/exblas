/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie 
 *  All rights reserved.
 */

#include "blas1.hpp"
#include "common.hpp"

#include <iostream>
#include <limits>
#include <string.h>

// All OpenCL headers
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

double ExDOTVsMPFR(int N, double *a, int inca, double *b, int incb) {
    mpfr_t sum, dot, op;
    mpfr_init2(op, 64);
    mpfr_init2(dot, 128);
    mpfr_init2(sum, 4196);

    mpfr_set_zero(dot, 0.0);
    mpfr_set_zero(sum, 0.0);

    for (int i = 0; i < N; i++) {
        mpfr_set_d(op, a[i], MPFR_RNDN);
        mpfr_mul_d(dot, op, b[i], MPFR_RNDN);
        mpfr_add(sum, sum, dot, MPFR_RNDN);
    }
    double dacc = mpfr_get_d(sum, MPFR_RNDN);

    mpfr_clear(op);
    mpfr_clear(dot);
    mpfr_clear(sum);
    mpfr_free_cache();

    return dacc;
}
#endif


int main(int argc, char *argv[]) {
    double eps = 1e-16;
    int N = 1 << 20;
    bool lognormal = false;
    if(argc > 1) {
        N = 1 << atoi(argv[1]);
    }
    if(argc > 4) {
        if(argv[4][0] == 'n') {
            lognormal = true;
        }
    }

    int range = 1;
    int emax = 0;
    double mean = 1., stddev = 1.;
    if(lognormal) {
        stddev = strtod(argv[2], 0);
        mean = strtod(argv[3], 0);
    }
    else {
        if(argc > 2) {
            range = atoi(argv[2]);
        }
        if(argc > 3) {
            emax = atoi(argv[3]);
        }
    }

    double *a, *b; 
    //a = (double*)_mm_malloc(N * sizeof(double), 32);
    int err = posix_memalign((void **) &a, 64, N * sizeof(double));
    err += posix_memalign((void **) &b, 64, N * sizeof(double));
    if ((!a) || (!b) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");
    if(lognormal) {
        init_lognormal(N, a, mean, stddev);
        init_lognormal(N, b, mean, stddev);
    } else if ((argc > 4) && (argv[4][0] == 'i')) {
        init_ill_cond(N, a, range);
        init_ill_cond(N, b, range);
    } else {
        if(range == 1){
            init_naive(N, a);
            init_naive(N, b);
        } else {
            init_fpuniform(N, a, range, emax);
            init_fpuniform(N, b, range, emax);
        }
    }

    fprintf(stderr, "%d ", N);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double exdot_acc, exdot_fpe3, exdot_fpe4, exdot_fpe8, exdot_fpe4ee, exdot_fpe6ee, exdot_fpe8ee;
    exdot_acc = exdot(N, a, 1, 0, b, 1, 0, 0);
    exdot_fpe3 = exdot(N, a, 1, 0, b, 1, 0, 3);
    exdot_fpe4 = exdot(N, a, 1, 0, b, 1, 0, 4);
    exdot_fpe8 = exdot(N, a, 1, 0, b, 1, 0, 8);
    exdot_fpe4ee = exdot(N, a, 1, 0, b, 1, 0, 4, true);
    exdot_fpe6ee = exdot(N, a, 1, 0, b, 1, 0, 6, true);
    exdot_fpe8ee = exdot(N, a, 1, 0, b, 1, 0, 8, true);
    printf("  exdot with superacc = %.16g\n", exdot_acc);
    printf("  exdot with FPE3 and superacc = %.16g\n", exdot_fpe3);
    printf("  exdot with FPE4 and superacc = %.16g\n", exdot_fpe4);
    printf("  exdot with FPE8 and superacc = %.16g\n", exdot_fpe8);
    printf("  exdot with FPE4 early-exit and superacc = %.16g\n", exdot_fpe4ee);
    printf("  exdot with FPE6 early-exit and superacc = %.16g\n", exdot_fpe6ee);
    printf("  exdot with FPE8 early-exit and superacc = %.16g\n", exdot_fpe8ee);


#ifdef EXBLAS_VS_MPFR
    double exdotMPFR = ExDOTVsMPFR(N, a, 1, b, 1);
    printf("  exdot with MPFR = %.16g\n", exdotMPFR);
    exdot_acc = fabs(exdotMPFR - exdot_acc) / fabs(exdotMPFR);
    exdot_fpe3 = fabs(exdotMPFR - exdot_fpe3) / fabs(exdotMPFR);
    exdot_fpe4 = fabs(exdotMPFR - exdot_fpe4) / fabs(exdotMPFR);
    exdot_fpe8 = fabs(exdotMPFR - exdot_fpe8) / fabs(exdotMPFR);
    exdot_fpe4ee = fabs(exdotMPFR - exdot_fpe4ee) / fabs(exdotMPFR);
    exdot_fpe6ee = fabs(exdotMPFR - exdot_fpe6ee) / fabs(exdotMPFR);
    exdot_fpe8ee = fabs(exdotMPFR - exdot_fpe8ee) / fabs(exdotMPFR);
    if ((exdot_acc > eps) || (exdot_fpe3 > eps) || (exdot_fpe4 > eps) || (exdot_fpe8 > eps) || (exdot_fpe4ee > eps) || (exdot_fpe6ee > eps) || (exdot_fpe8ee > eps)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g\n", exdot_acc, exdot_fpe3, exdot_fpe4, exdot_fpe8, exdot_fpe4ee, exdot_fpe6ee, exdot_fpe8ee);
    }
#else
    exdot_fpe3 = fabs(exdot_acc - exdot_fpe3) / fabs(exdot_acc);
    exdot_fpe4 = fabs(exdot_acc - exdot_fpe4) / fabs(exdot_acc);
    exdot_fpe8 = fabs(exdot_acc - exdot_fpe8) / fabs(exdot_acc);
    exdot_fpe4ee = fabs(exdot_acc - exdot_fpe4ee) / fabs(exdot_acc);
    exdot_fpe6ee = fabs(exdot_acc - exdot_fpe6ee) / fabs(exdot_acc);
    exdot_fpe8ee = fabs(exdot_acc - exdot_fpe8ee) / fabs(exdot_acc);
    if ((exdot_fpe3 > eps) || (exdot_fpe4 > eps) || (exdot_fpe8 > eps) || (exdot_fpe4ee > eps) || (exdot_fpe6ee > eps) || (exdot_fpe8ee > eps)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g\n", exdot_fpe3, exdot_fpe4, exdot_fpe8, exdot_fpe4ee, exdot_fpe6ee, exdot_fpe8ee);
    }
#endif
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

