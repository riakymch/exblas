/*
 * main.c
 *
 */
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <gmp.h>
#include <mpfr.h>
#include <math.h>

#include "common.hpp"

void usage(char * name) {
  printf("Usage: %s SIZE\n", name);
  printf("SIZE: size of input number to sum up\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc != 2)
    usage(argv[0]);
  int n = atoi(argv[1]);

  double *a = (double *) calloc(n * n, sizeof(double));
  double *b = (double *) calloc(n, sizeof(double));

  double c = 0;//20 * double(rand()) / double(RAND_MAX * .99) + 1.;

  //upper is row-wise
  int is_lower_column_wise = 1;
  generate_ill_cond_system(is_lower_column_wise, a, b, n, c);

  printMatrix(is_lower_column_wise, a, n, n);
  printVector(b, n);

  //printf("condA = %8.g\n", condA(a, n));

  return 0;
}

