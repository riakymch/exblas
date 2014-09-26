#ifndef OCLBLAS_S_H_
#define OCLBLAS_S_H_

#include "oclblas_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Level 3 BLAS functions
oclblas_err_t
oclblas_sgemm(
    const oclblas_order_t order,
    const oclblas_trans_t transa, const oclblas_trans_t transb,
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
oclblas_err_t
oclblas_sgemm0(
    const oclblas_order_t order,
    const oclblas_trans_t transa, const oclblas_trans_t transb,
    const int m, const int n, const int k,
    const float alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    const float beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_ssymm(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const int m, const int n,
    const float alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const float beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_ssymm0(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const int m, const int n,
    const float alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    float beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_ssyrk(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const float alpha,
    const cl_mem a, const int offseta, const int lda,
    const float beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_ssyrk0(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const float alpha,
    const cl_mem a, const int lda,
    const float beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_ssyr2k(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const float alpha,
    const cl_mem a, const int offseta, const int lda,
    const cl_mem b, const int offsetb, const int ldb,
    const float beta,
    cl_mem c, const int offsetc, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_ssyr2k0(
    const oclblas_order_t order,
    const oclblas_uplo_t uplo, const oclblas_trans_t trans,
    const int n, const int k,
    const float alpha,
    const cl_mem a, const int lda,
    const cl_mem b, const int ldb,
    const float beta,
    cl_mem c, const int ldc,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

oclblas_err_t
oclblas_strmm(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const oclblas_trans_t transa, const oclblas_diag_t diag,
    const int m, const int n,
    const float alpha,
    const cl_mem a, const int offseta, const int lda,
    cl_mem b, const int offsetb, const int ldb,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );
oclblas_err_t
oclblas_strmm0(
    const oclblas_order_t order,
    const oclblas_side_t side, const oclblas_uplo_t uplo,
    const oclblas_trans_t transa, const oclblas_diag_t diag,
    const int m, const int n,
    const float alpha,
    const cl_mem a, const int lda,
    cl_mem b, const int ldb,
    const cl_uint num_command_queues, cl_command_queue *command_queues,
    const cl_uint num_events_in_wait_list, const cl_event *event_wait_list,
    cl_event *events
    );

#ifdef __cplusplus
}
#endif


#endif // #ifndef OCLBLAS_S_H_
