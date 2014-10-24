
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <gmp.h>
#include <mpfr.h>
#include <math.h>

double randDouble(int emin, int emax, int neg_ratio);
void init_fpuniform(double *a, int n, int range, int emax);
void init_fpuniform_un_matrix(double *a, const uint n, const int range, const int emax);
char *sum_mpfr(double *data, int size);

