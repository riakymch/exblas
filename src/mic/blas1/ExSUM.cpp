/*
 *  Copyright (c) 2016 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <omp.h>

#include "ExSUM.hpp"
#include "blas1.hpp"

#ifdef EXBLAS_TIMING
    #define iterations 20
#endif


/*
 * Parallel summation using our algorithm
 * If fpe < 2, use superaccumulators only,
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
double exsum(int N, double *a, int inca, int offset, int fpe, bool early_exit) {
    if (fpe < 0) {
        fprintf(stderr, "Size of floating-point expansion should be a positive number. Preferably, it should be in the interval [2, 8]\n");
        exit(1);
    }

    // with superaccumulators only
    if (fpe < 2)
        return ExSUMSuperacc(N, a, inca);

    if (early_exit) {
        if (fpe <= 4)
            return (ExSUMFPE<FPExpansionVect<Vec8d, 4, true > >)(N, a, inca, offset);
        if (fpe <= 6)
            return (ExSUMFPE<FPExpansionVect<Vec8d, 6, true > >)(N, a, inca, offset);
        if (fpe <= 8)
            return (ExSUMFPE<FPExpansionVect<Vec8d, 8, true > >)(N, a, inca, offset);
    } else { // ! early_exit
        if (fpe == 2)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 2> >)(N, a, inca, offset);
        if (fpe == 3)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 3> >)(N, a, inca, offset);
        if (fpe == 4)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 4> >)(N, a, inca, offset);
        if (fpe == 5)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 5> >)(N, a, inca, offset);
        if (fpe == 6)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 6> >)(N, a, inca, offset);
        if (fpe == 7)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 7> >)(N, a, inca, offset);
        if (fpe == 8)
	   return (ExSUMFPE<FPExpansionVect<Vec8d, 8> >)(N, a, inca, offset);
    }

    return 0.0;
}

double ExSUMSuperaccBack(int N, double *a, int inca, int offset) {
    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
    	tstart = rdtsc();
#endif
	Superaccumulator sa(319, 300);

    	for(int i = 0; i != N; i += inca) {
            sa.Accumulate(a[i]);
        }

        sa.Normalize();
        dacc = sa.Round();
#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart)/N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}

double ExSUMSuperacc(int N, double *a, int inca, int offset) {
    int nthread = tbb::task_scheduler_init::automatic;
    tbb::task_scheduler_init tbbinit(nthread);

    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
    	tstart = rdtsc();
#endif
        TBBlongsum tbbsum(a);

        tbb::parallel_reduce(tbb::blocked_range<size_t>(0, N, inca), tbbsum);

        dacc = tbbsum.acc.Round();
#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart)/N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}

inline static void ReductionStep(int step, int tid1, int tid2, Superaccumulator *acc1, Superaccumulator *acc2, int volatile * ready1, int volatile * ready2) {
    int const delay = 16;
    _mm_prefetch((char const*)ready2, _MM_HINT_T0);
    acc2->Prefetch();
    // Wait for thread 2
    while(*ready2 < step) {
        // wait
        _mm_delay_32(delay);
    }
    acc1->Accumulate(*acc2);
}

template<typename CACHE> double ExSUMFPE(int N, double * a, int inca, int offset) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int const prefetch_distance_T0 = 1 * 16;
    int const prefetch_distance_T1 = 10 * 16;//7 * 16;
    int maxthreads = omp_get_max_threads();

    double dacc;
#ifdef EXBLAS_TIMING
    double t, mint = 10000;
    uint64_t tstart, tend;
    for(int iter = 0; iter != iterations; ++iter) {
#endif
        std::vector<Superaccumulator> acc(maxthreads);
        std::vector<int32_t> ready(maxthreads * linesize);
#ifdef EXBLAS_TIMING
        tstart = rdtsc();
#endif

        #pragma omp parallel
        {
            unsigned int tid = omp_get_thread_num();
            unsigned int tnum = omp_get_num_threads();

            CACHE cache(acc[tid]);
            *(int32_t volatile *)(&ready[tid * linesize]) = 0;

            int l = ((tid * int64_t(N)) / tnum) & ~15ul;
            int r = ((((tid+1) * int64_t(N)) / tnum) & ~15ul) - 1;

            _mm_prefetch((char const *)(a+l), _MM_HINT_T0);
            _mm_prefetch((char const *)(a+l+8), _MM_HINT_T0);

            cache.Accumulate(a+l, r-l);

            cache.Flush();
            acc[tid].Normalize();

            // Custom reduction
            for(int s = 1; (1 << (s-1)) < tnum; ++s) 
            {
                int32_t volatile * c = &ready[tid * linesize];
                ++*c;
                if(tid % (1 << s) == 0) {
                    int tid2 = tid | (1 << (s-1));
                    if(tid2 < tnum) {
                        ReductionStep(s, tid, tid2, &acc[tid], &acc[tid2],
                           &ready[tid * linesize], &ready[tid2 * linesize]);
                    }
                }
            }
        }

        dacc = acc[0].Round();
#ifdef EXBLAS_TIMING
        tend = rdtsc();
        t = double(tend - tstart) / N;
        mint = std::min(mint, t);
    }
    fprintf(stderr, "%f ", mint);
#endif

    return dacc;
}

