/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file blas1.hpp
 *  \brief Provides a set of Basic Linear Algera Subprograms level-1 to work with scalar-vectors
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef BLAS1_H_
#define BLAS1_H_

// config from cmake
#include "config.h"

/**
 * \defgroup blas1 BLAS Level-1 Functions
 */

/**
 * \defgroup ExSUM Summation Functions
 * \ingroup blas1
 */

/**
 * \ingroup ExSUM
 * \brief Parallel summation computes the sum of elements of a real vector with our 
 *     multi-level reproducible and accurate algorithm.
 *
 *     If fpe < 2, it uses superaccumulators only. Otherwise, it relies on 
 *     floating-point expansions of size FPE with superaccumulators when needed
 *
 * \param N vector size
 * \param a vector
 * \param inca specifies the increment for the elements of a
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double exsum(int N, double *a, int inca, int fpe, bool early_exit = false);

#endif // BLAS1_H_
