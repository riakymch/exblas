
#include "TRSV.hpp"

/*
 * Naive implementation of TRSV for comparision only; it is much easy to port than the BLAS implementation
 */
double TRSVCPU(
    double *a,
    double *b,
    const unsigned int n
) {
    double res = 0.0;
    for(unsigned int i = 0; i < n; i++)
        res += a[i] * b[i];

    return res;
}

extern "C" mpfr_t *TRSVWithMPFR(double *h_a, double *h_b, int size) {
  mpfr_t *sum, trsv, op1;
  int i;
  sum = (mpfr_t *) malloc(sizeof(mpfr_t));

  mpfr_init2(op1, 64);
  mpfr_init2(trsv, 128);
  mpfr_init2(*sum, 4196);

  mpfr_set_d(trsv, 0.0, MPFR_RNDN);
  mpfr_set_d(*sum, 0.0, MPFR_RNDN);

  for (i = 0; i < size; i++) {
    mpfr_set_d(op1, h_a[i], MPFR_RNDN);
    mpfr_mul_d(trsv, op1, h_b[i], MPFR_RNDN);
    mpfr_add(*sum, *sum, trsv, MPFR_RNDN);
  }

  mpfr_free_cache();

  return sum;
}

extern "C" bool CompareWithMPFR(mpfr_t *res_mpfr, double res_rounded) {
  double rounded_mpfr = mpfr_get_d(*res_mpfr, MPFR_RNDD);
  printf("GPU Parallel TRSV: %.15g\n", res_rounded);
  printf("Rounded value of MPFR: %.15g\n", rounded_mpfr);

  //Compare the results with MPFR using native functions
  bool res_cmp = false;
  if (abs(rounded_mpfr - res_rounded) < 1e-16){
      printf("\t The result is EXACT -- matches the MPFR algorithm!\n\n");
      res_cmp = true;
  } else {
      printf("\t The result is WRONG -- does not match the MPFR algorithm!\n\n");
  }

  mpfr_clear(*res_mpfr);
  free(res_mpfr);
  mpfr_free_cache();

  return res_cmp;
}

