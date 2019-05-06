/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

/**
 *  \file blas1.hpp
 *  \brief Provides a set of Basic Linear Algera Subprograms level-1 to handle scalar-vector operations
 *
 *  \authors
 *    Developers : \n
 *        Roman Iakymchuk  -- roman.iakymchuk@lip6.fr \n
 *        Sylvain Collange -- sylvain.collange@inria.fr \n
 */

#ifndef BLAS1_HPP_
#define BLAS1_HPP_

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
 * \param Ng vector size
 * \param ag vector
 * \param inca specifies the increment for the elements of a
 * \param offset specifies position in the vector from its start
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate sum of elements of a real vector
 */
double exsum(const int Ng, double *ag, const int inca, const int offset, const int fpe, const bool early_exit = false);

/**
 * \defgroup ExDOT Dot Product Functions
 * \ingroup blas1
 */

/**
 * \ingroup ExDOT
 * \brief Parallel dot forms the dot product of two vectors with our
 *     multi-level reproducible and accurate algorithm.
 *
 *     If fpe < 3, it uses superaccumulators only. Otherwise, it relies on 
 *     floating-point expansions of size FPE with superaccumulators when needed
 *
 * \param Ng vector size
 * \param ag vector
 * \param inca specifies the increment for the elements of a
 * \param offseta specifies position in the vector a from its start
 * \param bg vector
 * \param incb specifies the increment for the elements of b
 * \param offsetb specifies position in the vector b from its start
 * \param fpe stands for the floating-point expansions size (used in conjuction with superaccumulators)
 * \param early_exit specifies the optimization technique. By default, it is disabled
 * \return Contains the reproducible and accurate result of the dot product of two real vectors
 */
double exdot(const int Ng, double *ag, const int inca, const int offseta, double *bg, const int incb, const int offsetb, const int fpe, const bool early_exit = false);

#endif // BLAS1_HPP_

