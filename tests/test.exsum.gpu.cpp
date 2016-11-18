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

double ExSUMVsMPFR(int N, double *a) {
    mpfr_t mpaccum;
    mpfr_init2(mpaccum, 2098);
    mpfr_set_zero(mpaccum, 0);

    for(int i = 0; i != N; ++i) {
        mpfr_add_d(mpaccum, mpaccum, a[i], MPFR_RNDN);
    }
    double dacc = mpfr_get_d(mpaccum, MPFR_RNDN);

    //mpfr_printf("%Ra\n", mpaccum);
    mpfr_clear(mpaccum);

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

    double *a; 
    //a = (double*)_mm_malloc(N * sizeof(double), 32);
    int err = posix_memalign((void **) &a, 64, N * sizeof(double));
    if ((!a) || (err != 0))
        fprintf(stderr, "Cannot allocate memory with posix_memalign\n");
    if(lognormal) {
        init_lognormal(N, a, mean, stddev);
    } else if ((argc > 4) && (argv[4][0] == 'i')) {
        init_ill_cond(N, a, range);
    } else {
        if(range == 1){
            init_naive(N, a);
        } else {
            init_fpuniform(N, a, range, emax);
        }
    }

    fprintf(stderr, "%d ", N);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double exsum_acc, exsum_fpe2, exsum_fpe3, exsum_fpe4, exsum_fpe8, exsum_fpe4ee, exsum_fpe6ee, exsum_fpe8ee;
    exsum_acc = exsum(N, a, 1, 0, 0);
    exsum_fpe2 = exsum(N, a, 1, 0, 2);
    exsum_fpe3 = exsum(N, a, 1, 0, 3);
    exsum_fpe4 = exsum(N, a, 1, 0, 4);
    exsum_fpe8 = exsum(N, a, 1, 0, 8);
    exsum_fpe4ee = exsum(N, a, 1, 0, 4, true);
    exsum_fpe6ee = exsum(N, a, 1, 0, 6, true);
    exsum_fpe8ee = exsum(N, a, 1, 0, 8, true);
    printf("  exsum with superacc = %.16g\n", exsum_acc);
    printf("  exsum with FPE2 and superacc = %.16g\n", exsum_fpe2);
    printf("  exsum with FPE3 and superacc = %.16g\n", exsum_fpe3);
    printf("  exsum with FPE4 and superacc = %.16g\n", exsum_fpe4);
    printf("  exsum with FPE8 and superacc = %.16g\n", exsum_fpe8);
    printf("  exsum with FPE4 early-exit and superacc = %.16g\n", exsum_fpe4ee);
    printf("  exsum with FPE6 early-exit and superacc = %.16g\n", exsum_fpe6ee);
    printf("  exsum with FPE8 early-exit and superacc = %.16g\n", exsum_fpe8ee);

#ifdef EXBLAS_VS_MPFR
    double exsumMPFR = ExSUMVsMPFR(N, a);
    printf("  exsum with MPFR = %.16g\n", exsumMPFR);
    exsum_acc = fabs(exsumMPFR - exsum_acc) / fabs(exsumMPFR);
    exsum_fpe2 = fabs(exsumMPFR - exsum_fpe2) / fabs(exsumMPFR);
    exsum_fpe3 = fabs(exsumMPFR - exsum_fpe3) / fabs(exsumMPFR);
    exsum_fpe4 = fabs(exsumMPFR - exsum_fpe4) / fabs(exsumMPFR);
    exsum_fpe8 = fabs(exsumMPFR - exsum_fpe8) / fabs(exsumMPFR);
    exsum_fpe4ee = fabs(exsumMPFR - exsum_fpe4ee) / fabs(exsumMPFR);
    exsum_fpe6ee = fabs(exsumMPFR - exsum_fpe6ee) / fabs(exsumMPFR);
    exsum_fpe8ee = fabs(exsumMPFR - exsum_fpe8ee) / fabs(exsumMPFR);
    if ((exsum_acc > eps) || (exsum_fpe2 > eps) || (exsum_fpe4 > eps) || (exsum_fpe3 > eps) || (exsum_fpe8 > eps) || (exsum_fpe4ee > eps) || (exsum_fpe6ee > eps) || (exsum_fpe8ee > eps)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g\n", exsum_acc, exsum_fpe2, exsum_fpe3, exsum_fpe4, exsum_fpe8, exsum_fpe4ee, exsum_fpe6ee, exsum_fpe8ee);
    }
#else
    exsum_fpe2 = fabs(exsum_acc - exsum_fpe2) / fabs(exsum_acc);
    exsum_fpe3 = fabs(exsum_acc - exsum_fpe3) / fabs(exsum_acc);
    exsum_fpe4 = fabs(exsum_acc - exsum_fpe4) / fabs(exsum_acc);
    exsum_fpe8 = fabs(exsum_acc - exsum_fpe8) / fabs(exsum_acc);
    exsum_fpe4ee = fabs(exsum_acc - exsum_fpe4ee) / fabs(exsum_acc);
    exsum_fpe6ee = fabs(exsum_acc - exsum_fpe6ee) / fabs(exsum_acc);
    exsum_fpe8ee = fabs(exsum_acc - exsum_fpe8ee) / fabs(exsum_acc);
    if ((exsum_fpe2 > eps) || (exsum_fpe4 > eps) || (exsum_fpe3 > eps) || (exsum_fpe8 > eps) || (exsum_fpe4ee > eps) || (exsum_fpe6ee > eps) || (exsum_fpe8ee > eps)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g \t %.16g\n", exsum_fpe2, exsum_fpe3, exsum_fpe4, exsum_fpe8, exsum_fpe4ee, exsum_fpe6ee, exsum_fpe8ee);
    }
#endif
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

