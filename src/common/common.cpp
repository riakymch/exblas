/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cstdlib>
#include <cstdio>
#include <random>
#include <math.h>
#include "common.hpp"


double randDoubleUniform() {
    // Uniform distribution for now
    return double(rand() - RAND_MAX/4) * 12345.678901234;
}

double randDouble(int emin, int emax, int neg_ratio) {
    // Uniform mantissa
    double x = double(rand()) / double(RAND_MAX * .99) + 1.;
    // Uniform exponent
    int e = (rand() % (emax - emin)) + emin;
    // Sign
    if(neg_ratio > 1 && rand() % neg_ratio == 0) {
        x = -x;
    }
    return ldexp(x, e);
}

void init_fpuniform(const int n, double *a, int range, int emax) {
    for(int i = 0; i != n; ++i)
        a[i] = randDouble(emax-range, emax, 1);
}

void init_fpuniform_matrix(const bool iscolumnwise, const int m, const int n, double *a, const int lda, const int range, const int emax) {
    //Generate numbers on several bins starting from emax
    if (iscolumnwise) {
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                a[j * lda + i] = randDouble(0, range, 1);
    } else {
        for(int i = 0; i < m; ++i)
            for(int j = 0; j < n; ++j)
                a[i * lda + j] = randDouble(0, range, 1);
    }
}

void init_fpuniform_tr_matrix(const char uplo, const char diag, const int n, double *a, const int range, const int emax) {
    if (uplo == 'U') {
        for(int i = n-1; i >= 0; i--)
            for(int j = i; j < n; ++j)
                if ((diag == 'U') && (j == i))
                    a[j * n + i] = 1.0;
                else
                    a[j * n + i] = randDouble(emax-range, emax, 1);
    } else {
        for(int i = 0; i != n; ++i)
            for(int j = 0; j <= i; ++j)
                if ((diag == 'U') && (j == i))
                    a[j * n + i] = 1.0;
                else
                    a[j * n + i] = randDouble(emax-range, emax, 1);
    }
}

void init_lognormal(const int n, double * a, double mean, double stddev) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::lognormal_distribution<> d(mean, stddev);

    for(int i = 0; i != n; ++i)
        a[i] = d(gen);
}

void init_lognormal_matrix(const bool iscolumnwise, const int m, const int n, double *a, const int lda, const double mean, const double stddev) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::lognormal_distribution<> d(mean, stddev);

    if (iscolumnwise) {
        for(int j = 0; j < n; ++j)
            for(int i = 0; i < m; ++i)
                a[j * lda + i] = 1.0;//d(gen);
    } else {
        for(int i = 0; i < m; ++i)
            for(int j = 0; j < n; ++j)
                a[i * lda + j] = 1.0;//d(gen);
    }
}

void init_lognormal_tr_matrix(const char uplo, const char diag, const int n, double * a, const double mean, const double stddev) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::lognormal_distribution<> d(mean, stddev);

    if (uplo == 'U') {
        for(int i = n-1; i >= 0; i--)
            for(int j = i; j < n; ++j)
                if ((diag == 'U') && (j == i))
                    a[i * n + j] = 1.0;
                else
                    a[i * n + j] = d(gen);
    } else {
        for(int i = 0; i != n; ++i)
            for(int j = 0; j <= i; ++j)
                if ((diag == 'U') && (j == i))
                    a[i * n + j] = 1.0;
                else
                    a[i * n + j] = d(gen);
    }
}

void init_ill_cond(const int n, double *a, double c) {
    int n2 = round(n / 2);

    for(int i = 0; i != n; ++i)
        a[i] = 0.0;

    // init the first half of exponents
    double *e, x;
    e = (double *) malloc(n * sizeof(double));
    double b = log2(c);
    for(int i = 0; i != n2; ++i) {
        x = double(rand()) / double(RAND_MAX);
        e[i] = round(x * b / 2);
    }
    e[0] = round(b / 2) + 1.;
    e[n-1] = 0;

    // init the first half of the vector
    for(int i = 0; i != n2; ++i) {
        double x = double(rand()) / double(RAND_MAX);
        a[i] = (2. * x - 1.) * pow(2., e[i]);
    }

    // init the second half of the vector
    double step = (b / 2) / (n - n2);
    for(int i = n2; i != n; ++i) {
        double x = double(rand()) / double(RAND_MAX);
        e[i] = step * (i - n2);
        a[i] = (2. * x - 1.) * pow(2., e[i]); 
    }

    free(e);
}

void init_naive(const int n, double *a) {
    for(int i = 0; i != n; ++i)
        a[i] = 1.1;
}

