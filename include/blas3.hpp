/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file blas3.hpp
 *  \brief Provides a set of Basic Linear Algera Subprograms level-3 to work with matrices
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef BLAS3_H_
#define BLAS3_H_

// config from cmake
#include "config.h"

/**
 * \defgroup blas3 BLAS Level-3 Functions
 */

/**
 * \defgroup xgemm GEMM Functions
 * \ingroup blas3
 */

/**
 * \ingroup xgemm
 * \brief ExGEMM computes the matrix-matrix multiplication, which are composed of real numbers, using our 
 *     multi-level reproducible and accurate algorithm.
 *
 *     If fpe < 2, it uses superaccumulators only.
 *     Otherwise, it relies on floating-point expansions of size FPE with superaccumulators when needed
 *
 * \param N matrix size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double dsum(int N, double *a, int inca, int fpe, bool early_exit = false);

#endif // BLAS3_H_
