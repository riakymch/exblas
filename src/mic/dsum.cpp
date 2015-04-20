/*
 *  Copyright (c) 2013-2015 Inria and University Pierre and Marie Curie
 *  All rights reserved.
 */

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <omp.h>

#include "dsum.hpp"
#include "blas1.hpp"

/*
 * Parallel summation using our algorithm
 * If fpe < 2, use superaccumulators only,
 * Otherwise, use floating-point expansions of size FPE with superaccumulators when needed
 * early_exit corresponds to the early-exit technique
 */
double dsum(int N, double *a, int inca, int fpe, bool early_exit) {
    if (fpe < 0) {
	fprintf(stderr, "Size of floating-point expansion should be a positive number. Preferably, it should be in the interval [2, 8]\n");
        exit(1);
    }

    // with superaccumulators only
    if (fpe < 2)
        return xTBBSuperacc(N, a, inca);
    // there is no need and no improvement at all in using the early-exit technique for FPE of size 2
    if (fpe == 2)
        return (xOMPFPE<FPExpansionVect<Vec8d, 2> >)(N, a, inca);
   
    if (early_exit) {
        if (fpe == 3)
	    return (xOMPFPE<FPExpansionVect<Vec8d, 3, true > >)(N, a, inca);
        if (fpe == 4)
	    return (xOMPFPE<FPExpansionVect<Vec8d, 4, true > >)(N, a, inca);
        if (fpe == 5)
	    return (xOMPFPE<FPExpansionVect<Vec8d, 5, true > >)(N, a, inca);
        if (fpe == 6)
	    return (xOMPFPE<FPExpansionVect<Vec8d, 6, true > >)(N, a, inca);
        if (fpe == 7)
	    return (xOMPFPE<FPExpansionVect<Vec8d, 7, true > >)(N, a, inca);
        if (fpe == 8)
	    return (xOMPFPE<FPExpansionVect<Vec8d, 8, true > >)(N, a, inca);
    } else { // ! early_exit
        if (fpe == 3)
            return (xOMPFPE<FPExpansionVect<Vec8d, 3> >)(N, a, inca);
        if (fpe == 4)
            return (xOMPFPE<FPExpansionVect<Vec8d, 4> >)(N, a, inca);
        if (fpe == 5)
            return (xOMPFPE<FPExpansionVect<Vec8d, 5> >)(N, a, inca);
        if (fpe == 6)
            return (xOMPFPE<FPExpansionVect<Vec8d, 6> >)(N, a, inca);
        if (fpe == 7)
            return (xOMPFPE<FPExpansionVect<Vec8d, 7> >)(N, a, inca);
        if (fpe == 8)
            return (xOMPFPE<FPExpansionVect<Vec8d, 8> >)(N, a, inca);
    }
    
    return 0.0;
}

double xSuperacc(int N, double *a, int inca) {
#ifdef EXBLAS_TIMING
    uint64_t tstart = rdtsc();
#endif
    Superaccumulator sa(319, 300);

    for(int i = 0; i != N; i += inca) {
        sa.Accumulate(a[i]);
    }

    sa.Normalize();
    double dacc = sa.Round();
#ifdef EXBLAS_TIMING
    uint64_t tend = rdtsc();
    double t = double(tend - tstart)/N;
    printf("time = %f (%f Gacc)\n", t, freq/t);
#endif

    return dacc;
}

double xTBBSuperacc(int N, double *a, int inca) {
    int nthread = tbb::task_scheduler_init::automatic;
    tbb::task_scheduler_init tbbinit(nthread);

#ifdef EXBLAS_TIMING
    uint64_t tstart = rdtsc();
#endif
    TBBlongsum tbbsum(a);

    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, N, inca), tbbsum);

    double dacc = tbbsum.acc.Round();
#ifdef EXBLAS_TIMING
    uint64_t tend = rdtsc();
    double t = double(tend - tstart)/N;
    printf("time = %f (%f Gacc)\n", t, freq/t);
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

template<typename CACHE> double xOMPFPE(int N, double * a, int inca) {
    // OpenMP sum+reduction
    int const linesize = 16;    // * sizeof(int32_t)
    int const prefetch_distance_T0 = 1 * 16;
    int const prefetch_distance_T1 = 10 * 16;//7 * 16;
    int maxthreads = omp_get_max_threads();
    std::vector<Superaccumulator> acc(maxthreads);
    std::vector<int32_t> ready(maxthreads * linesize);
#ifdef EXBLAS_TIMING
    uint64_t tstart = rdtsc();
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

    double dacc = acc[0].Round();
#ifdef EXBLAS_TIMING
    uint64_t tend = rdtsc();
    double t = double(tend - tstart)/N;
    printf("time = %f (%f Gacc)\n", t, freq/t);
#endif

    return dacc;
}
