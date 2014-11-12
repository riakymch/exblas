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

  double c = 10;//20 * double(rand()) / double(RAND_MAX * .99) + 1.;
  printf("c = %.4g\n", c);

  generate_ill_cond_system(a, b, n, c);

  //printf("condA = %8.g\n", condA(a, n));

  return 0;
}

