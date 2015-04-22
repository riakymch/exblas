/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
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

void init_fpuniform(double *a, int n, int range, int emax) {
    for(int i = 0; i != n; ++i) {
        a[i] = randDouble(emax-range, emax, 1);
        //a[i] = randDoubleUniform();
        //printf("%a ", a[i]);
    }
}

void init_fpuniform(double *a, int m, int n, int range, int emax) {
}

void init_lognormal(double * a, int n, double mean, double stddev) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::lognormal_distribution<> d(mean, stddev);

    printf("min=%a, max=%a\n", d.min(), d.max());

    for(int i = 0; i != n; ++i) {
        a[i] = d(gen);
    }
}

void init_lognormal(double *a, int m, int n, double mean, double stddev) {
}

void init_ill_cond(double *a, int n, double c) {
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

void init_ill_cond(double *a, int m, int n, double c) {
}

void init_naive(double *a, int n) {
    for(int i = 0; i != n; ++i)
        a[i] = 1.0;
}

void init_naive(double *a, int m, int n) {
    for(int j = 0; j != n; ++j)
        for(int i = 0; i != m; ++i)
            a[j * m + i] = 1.0;
}
