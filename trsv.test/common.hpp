/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string.h>

////////////////////////////////////////////////////////////////////////////////
// Common definitions
////////////////////////////////////////////////////////////////////////////////
#define BIN_COUNT 76U


#define E_BITS 2 * 1023
#define F_BITS 2 * (1023 + 52)


////////////////////////////////////////////////////////////////////////////////
// Common functions
////////////////////////////////////////////////////////////////////////////////
void init_fpuniform(
    double *a,
    const uint n,
    const int range,
    const int emax
);

void init_fpuniform_lu_matrix(
    double *a,
    const uint n,
    const int range,
    const int emax
);

void init_fpuniform_un_matrix(
    double *a,
    const uint n,
    const int range,
    const int emax
);

void generate_ill_cond_system(
    double *a,
    double *b,
    const int n,
    const double c
);

double min(
    double arr[],
    int size
);

extern "C" double TwoProductFMA(
    double a,
    double b,
    double *d
);

extern "C" double KnuthTwoSum(
    double a,
    double b,
    double *d
);

#endif
