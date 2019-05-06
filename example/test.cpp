/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "blas1.hpp"
#include "common.hpp"

#ifdef EXBLAS_VS_MPFR
#include <cstddef>
#include <mpfr.h>

double exsumVsMPFR(int N, double *a) {
    mpfr_t mpaccum;
    mpfr_init2(mpaccum, 2098);
    mpfr_set_zero(mpaccum, 0);

    for(int i = 0; i != N; ++i) {
        mpfr_add_d(mpaccum, mpaccum, a[i], MPFR_RNDN);
    }
    double dacc = mpfr_get_d(mpaccum, MPFR_RNDN);

    mpfr_clear(mpaccum);

    return dacc;
}
#endif


int main(int argc, char * argv[]) {
    int N = 1 << 24;
    if(argc > 1) {
        N = 1 << atoi(argv[1]);
    }

    bool lognormal = false;
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
    int err = posix_memalign((void **) &a, 64, N * sizeof(double));
    if ((!a) || (err != 0))
        fprintf(stderr, "Cannot allocate memory for the main array\n");
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
    double exsum_fpe2, exsum_fpe4, exsum_fpe8ee;
    exsum_fpe2 = exsum(N, a, 1, 2);
    exsum_fpe4 = exsum(N, a, 1, 4);
    exsum_fpe8ee = exsum(N, a, 1, 8, true);
    printf("  exsum with FPE2 and superacc = %.16g\n", exsum_fpe2);
    printf("  exsum with FPE4 and superacc = %.16g\n", exsum_fpe4);
    printf("  exsum with FPE8 early-exit and superacc = %.16g\n", exsum_fpe8ee);

#ifdef EXBLAS_VS_MPFR
    double exsumMPFR = exsumVsMPFR(N, a);
    printf("  exsum with MPFR = %.16g\n", exsumMPFR);
    if ((fabs(exsumMPFR - exsum_fpe2) != 0) || (fabs(exsumMPFR - exsum_fpe4) != 0) || (fabs(exsumMPFR - exsum_fpe8ee) != 0)) {
        is_pass = false;
        printf("FAILED: %.16g \t %.16g \t %.16g\n", fabs(exsumMPFR - exsum_fpe2), fabs(exsumMPFR - exsum_fpe4), fabs(exsumMPFR - exsum_fpe8ee));
    }
#endif

    fprintf(stderr, "\n");

    if (is_pass)
        printf("TestPassed; ALL OK!\n");
    else
        printf("TestFailed!\n");

    return 0;
}
