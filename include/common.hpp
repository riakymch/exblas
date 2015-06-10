/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file common.hpp
 *  \brief A set of common definitions and functions
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * \defgroup common Common Definitions and Functions
 * \ingroup blas1
 */

/**
 * \ingroup common
 * \brief Maximum exponent
 */
int constexpr e_bits = 1023;    // 319

/**
 * \ingroup common
 * \brief Maximum exponent + the number of bits in signigicant
 */
int constexpr f_bits = 1023 + 52;   // 300

/**
 * \ingroup common
 * \brief Number of limbs in superaccumulator
 */
int constexpr bin_count = 39;


/**
 * \ingroup common
 * \brief Generates a random number for log-uniform distribution
 *
 * \return The generated number
 */
double randDoubleUniform();

/**
 * \ingroup common
 * \brief Generates a random number for uniform distribution
 *
 * \param emin minimum exponent
 * \param emax maximum exponenet
 * \param neg_ratio
 * \return The generated number
 */
double randDouble(int emin, int emax, int neg_ratio);

/**
 * \ingroup common
 * \brief Generates a real vector with uniform distribution of ellements
 *
 * \param a input/output vector
 * \param n vector size
 * \param range dynamic range of generated elements
 * \param emax maximum exponent (emax + range < EMAX)
 */
void init_fpuniform(double *a, int n, int range, int emax);

/**
 * \ingroup common
 * \brief Generates a real lower/upper unit/non-unit matrix with uniform distribution of ellements
 *
 * \param uplo L|U lower or upper triangular matrix
 * \param diag U|N unit or non-unit diagonal
 * \param a input/output matrix
 * \param n matrix size
 * \param range dynamic range of generated elements
 * \param emax maximum exponent (emax + range < EMAX)
 */
void init_fpuniform_matrix(char uplo, char diag, double *a, int n, int range, int emax);

/**
 * \ingroup common
 * \brief Generates a real vector with log-uniform distribution of ellements
 *
 * \param a input/output vector
 * \param n vector size
 * \param mean 
 * \param stddev
 */
void init_lognormal(double *a, int n, double mean, double stddev);

/**
 * \ingroup common
 * \brief Generates a real lower/upper unit/non-unit matrux with log-uniform distribution of ellements
 *
 * \param uplo L|U lower or upper triangular matrix
 * \param diag U|N unit or non-unit diagonal
 * \param a input/output matrix
 * \param n matrix size
 * \param mean
 * \param stddev
 */
void init_lognormal_matrix(char uplo, char diag, double *a, int n, double mean, double stddev);

/**
 * \ingroup common
 * \brief Generates a real vector with ill-conditioned ellements using \n
 * Algorithm 6.1. Generation of extremely ill-conditioned dot products from \n
 * T.Ogita et al. Accurate sum and dot product, SIAM Journal on Scientific Computing, 
 * 26(6):1955-1988, 2005.
 *
 * \param a input/output vector
 * \param n vector size
 * \param c anticipated condition number
 */
void init_ill_cond(double *a, int n, double c);

/**
 * \ingroup common
 * \brief Generates a real vector with all ellements set to 1.1
 * \param a input/output vector
 * \param n vector size
 */
void init_naive(double *a, int n);

#endif // COMMON_H
