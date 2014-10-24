
#include "common.hpp"

void init_fpuniform(double *a, int n, int range, int emax) {
  for (int i = 0; i != n; ++i) {
    //a[i] = randDouble(emax - range, emax, 1);
    a[i] = randDouble(0, range, 1);
  }
}

void init_fpuniform_un_matrix(double *a, const uint n, const int range, const int emax)
{
    //Generate numbers on several bins starting from emax
    for(uint i = 0; i < n; ++i)
        for(uint j = 0; j < n; ++j)
            if (j >= i)
                a[i * n + j] = randDouble(0, range, 1);
            else
                a[i * n + j] = 0.0;
}

double randDouble(int emin, int emax, int neg_ratio) {
  // Uniform mantissa
  double x = double(rand()) / double(RAND_MAX * .99) + 1.;
  // Uniform exponent
  int e = (rand() % (emax - emin)) + emin;
  // Sign
  if (neg_ratio > 1 && rand() % neg_ratio == 0) {
    x = -x;
  }
  return ldexp(x, e);
}

char *sum_mpfr(double *data, int size) {
  mpfr_t result;
  int i;

  mpfr_init2(result, 2098);
  mpfr_set_d(result, 0.0, MPFR_RNDN);

  for (i = 0; i < size; i++)
    mpfr_add_d(result, result, data[i], MPFR_RNDN);

  //printf ("\tSum MPFR (52):");
  //mpfr_out_str (stdout, 10, 52, result, MPFR_RNDD);
  //putchar ('\n');
  mpfr_exp_t exp_ptr;
  char *res_str = mpfr_get_str(NULL, &exp_ptr, 10, 52, result, MPFR_RNDD);
  printf("\tSum MPFR (52)      : %s \t e%d\n", res_str, (int) exp_ptr);
  mpfr_free_str(res_str);

  //mpfr_out_str (stdout, 2, 0, result, MPFR_RNDD);
  char *res = mpfr_get_str(NULL, &exp_ptr, 10, 2098, result, MPFR_RNDD);
  printf("\tSum MPFR (2098)    : %s\n", res);

  mpfr_clear(result);
  mpfr_free_cache();

  return res;
}

