#ifndef OCLBLAS_INTERNAL_INTERFACE_H_
#define OCLBLAS_INTERNAL_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (double)abs(x)
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (double)min(a,b)
#define dmax(a,b) (double)max(a,b)
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((unsigned int)1 << (b)))
#define bit_set(a,b)	((a) |  ((unsigned int)1 << (b)))

cl_kernel oclblas_get_kernel(const char *kernel_name);
cl_mem oclblas_get_bufa(const size_t required_bufa_size);
cl_mem oclblas_get_bufb(const size_t required_bufb_size);

#define OCLBLAS_TRANS   (0)
#define OCLBLAS_NOTRANS (1)

oclblas_err_t
oclblas_dgemm_internal(
    const int nota, const int notb,
    const int m, const int n, const int k,
    const double alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const double beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_sgemm_internal(
    const int nota, const int notb,
    const int m, const int n, const int k,
    const float alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const float beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

// TODO
#define GEMM_ALGO_MM (0)
#define GEMM_ALGO_MP (1)
#define GEMM_ALGO_PM (2)
#define GEMM_ALGO_PP (3)
//#include "algo.h"
#define GEMM_ALGO GEMM_ALGO_MM
#define GEMM_BSZ (384)

#ifdef __cplusplus
}
#endif

#endif // OCLBLAS_INTERNAL_INTERFACE_H_
