/*
 * main.c
 *
 *  Created on: Nov 20, 2013
 *      Author: riakym
 */
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <gmp.h>
#include <mpfr.h>
#include <math.h>

#include "common.hpp"
#include "Superaccumulator.hpp"
#include "TRSV.hpp"

void usage(char * name) {
  printf("Usage: %s SIZE\n", name);
  printf("SIZE: size of input number to sum up\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 2)
    usage(argv[0]);
  int n = atoi(argv[1]);

  double *a = (double *) malloc(n * n * sizeof(double));
  double *b = (double *) malloc(n * sizeof(double));
  double *x = (double *) malloc(n * sizeof(double));
  init_fpuniform_un_matrix(a, n, 1, 1023);
  init_fpuniform(b, n, 1, 1023);

  TRSVUNN(a, b, n, x);
  //TRSVUNN_Kulisch(a, b, n, x);

  //verify trsv and compute error
  //double error = verifyTRSVUNN(a, b, x, n, 1e-16);
  //printf("error = %8.g\n", error);
  compareTRSVUNNToMPFR(a, b, x, n, 1e-16);

  printf("condA = %8.g\n", condA(a, n));

  //compute condA

  return 0;
}

