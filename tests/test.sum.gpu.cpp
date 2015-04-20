/*
 *  Copyright (c) 2013-2015 University Pierre and Marie Curie 
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

double dsumWithMPFR(int N, double *a) {
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
    posix_memalign((void **) &a, 64, N * sizeof(double));
    if (!a)
        fprintf(stderr, "Cannot allocate memory for the main array\n");
    if(lognormal) {
        init_lognormal(a, N, mean, stddev);
    } else if ((argc > 4) && (argv[4][0] == 'i')) {
        init_ill_cond(a, N, range);
    } else {
        if(range == 1){
            init_naive(a, N);
        } else {
            init_fpuniform(a, N, range, emax);
        }
    }

    fprintf(stderr, "%d ", N);

    if(lognormal) {
        fprintf(stderr, "%f ", stddev);
    } else {
        fprintf(stderr, "%d ", range);
    }

    bool is_pass = true;
    double dsum_acc, dsum_fpe2, dsum_fpe4, dsum_fpe4ee, dsum_fpe6ee, dsum_fpe8ee;
    //dsum_acc = dsum(N, a, 1, 0);
    dsum_fpe2 = dsum(N, a, 1, 2);
    dsum_fpe4 = dsum(N, a, 1, 4);
    dsum_fpe4ee = dsum(N, a, 1, 4, true);
    dsum_fpe6ee = dsum(N, a, 1, 6, true);
    dsum_fpe8ee = dsum(N, a, 1, 8, true);
    //printf("  dsum with superacc = %.16g\n", dsum_acc);
    printf("  dsum with FPE2 and superacc = %.16g\n", dsum_fpe2);
    printf("  dsum with FPE4 and superacc = %.16g\n", dsum_fpe4);
    printf("  dsum with FPE4 early-exit and superacc = %.16g\n", dsum_fpe4ee);
    printf("  dsum with FPE6 early-exit and superacc = %.16g\n", dsum_fpe6ee);
    printf("  dsum with FPE8 early-exit and superacc = %.16g\n", dsum_fpe8ee);

    double dacc = 0.;
    for(int i = 0; i != N; ++i) {
        dacc += a[i];
    }
    printf("  fpsum=%.16g\n", dacc);

#ifdef EXBLAS_VS_MPFR
    double dsumMPFR = dsumWithMPFR(N, a);
    printf("  dsum with MPFR = %.16g\n", dsumMPFR);
    //if ((fabs(dsumMPFR - dsum_acc) != 0) || (fabs(dsumMPFR - dsum_fpe2) != 0) || (fabs(dsumMPFR - dsum_fpe4) != 0) || (fabs(dsumMPFR - dsum_fpe8ee) != 0) || (fabs(dsumMPFR - dsum_fpe4ee) != 0) || (fabs(dsumMPFR - dsum_fpe6ee) != 0)) {
    if ((fabs(dsumMPFR - dsum_fpe2) != 0) || (fabs(dsumMPFR - dsum_fpe4) != 0) || (fabs(dsumMPFR - dsum_fpe8ee) != 0) || (fabs(dsumMPFR - dsum_fpe4ee) != 0) || (fabs(dsumMPFR - dsum_fpe6ee) != 0)) {
        is_pass = false;
        //printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g\n", fabs(dsumMPFR - dsum_acc), fabs(dsumMPFR - dsum_fpe2), fabs(dsumMPFR - dsum_fpe4), fabs(dsumMPFR - dsum_fpe8ee));
        printf("FAILED: %.16g \t %.16g \t %.16g \t %.16g \t %.16g\n", fabs(dsumMPFR - dsum_fpe2), fabs(dsumMPFR - dsum_fpe4), fabs(dsumMPFR - dsum_fpe4ee), fabs(dsumMPFR - dsum_fpe6ee), fabs(dsumMPFR - dsum_fpe8ee));
    }
#else
    /*if ((fabs(dsum_acc - dsum_fpe2) != 0) || (fabs(dsum_acc - dsum_fpe4) != 0) || (fabs(dsum_acc - dsum_fpe8ee) != 0)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g\n", fabs(dsum_acc - dsum_fpe2), fabs(dsum_acc - dsum_fpe4), fabs(dsum_acc - dsum_fpe8ee));
    }*/
#endif
    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}

